"""
Character mapper for speaker identification.
Maps audio clusters to character names based on confident LLM assignments.
"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict

from ..utils.logger_utils import setup_logging
from ..narrative_storage_management.narrative_models import DialogueLine

logger = setup_logging(__name__)

class CharacterMapper:
    def map_audio_clusters_to_characters(self, dialogue_lines: List[DialogueLine]) -> Dict[str, str]:
        """Maps audio speaker IDs to character names based on confident LLM assignments.
        
        The workflow is:
        1. Audio diarization assigns speaker IDs (SPEAKER_06, SPEAKER_07, etc.)
        2. LLM identifies some dialogues with confident character names
        3. We map speaker IDs to character names using the confident LLM assignments
        """
        speaker_to_character_votes = defaultdict(lambda: defaultdict(int))
        
        # First pass: collect votes from confident LLM assignments
        for dialogue in dialogue_lines:
            if dialogue.is_llm_confident and dialogue.characters and dialogue.resolution_method == "audio_diarization":
                # This dialogue has speaker IDs from diarization and LLM confidence
                # All speakers get equal votes (no primary/secondary distinction)
                if hasattr(dialogue, 'original_llm_speaker') and dialogue.original_llm_speaker:
                    character_name = dialogue.original_llm_speaker
                    
                    # Give equal votes to all speakers
                    for speaker_id in dialogue.characters:
                        speaker_to_character_votes[speaker_id][character_name] += 1.0  # Full vote for all speakers
                        logger.debug(f"🗳️ Equal vote: Speaker '{speaker_id}' → Character '{character_name}' (dialogue {dialogue.index})")
        
        # Second pass: also check dialogues that have LLM confidence but no diarization assignment
        for dialogue in dialogue_lines:
            if dialogue.is_llm_confident and dialogue.speaker and dialogue.resolution_method != "audio_diarization":
                # This dialogue has LLM confidence but no diarization speaker ID
                # We can't map it directly, but we can use it for character name validation
                character_name = dialogue.speaker
                logger.debug(f"🗳️ LLM confident dialogue {dialogue.index}: '{character_name}' (no diarization mapping)")
        
        mapping = {}
        for speaker_id, character_votes in speaker_to_character_votes.items():
            if character_votes:
                # Select the character with the most votes for this speaker
                most_voted_character = max(character_votes, key=character_votes.get)
                mapping[speaker_id] = most_voted_character
                logger.info(f"🗳️ Mapped audio speaker '{speaker_id}' to character '{most_voted_character}' based on {character_votes[most_voted_character]} confident votes.")
            else:
                logger.warning(f"⚠️ No character mapping found for speaker '{speaker_id}'")
        
        return mapping

    def propagate_assignments(self, dialogue_lines: List[DialogueLine], speaker_mapping: Dict[str, str]) -> List[DialogueLine]:
        """Propagates confident assignments to all dialogues with the same speaker ID.
        Since we now have direct speaker assignments, we propagate based on speaker ID matching.
        """
        logger.info("Starting propagation of assignments.")
        for i, dialogue in enumerate(dialogue_lines):
            # Check if this dialogue has speaker IDs from diarization that need to be mapped to character names
            if dialogue.characters and dialogue.resolution_method == "audio_diarization":
                # Map all speakers to character names
                mapped_characters = []
                for speaker_id in dialogue.characters:
                    if speaker_id in speaker_mapping:
                        character_name = speaker_mapping[speaker_id]
                        mapped_characters.append(character_name)
                        logger.debug(f"➡️ Mapped speaker '{speaker_id}' to character '{character_name}' for dialogue line {i+1}.")
                    else:
                        logger.debug(f"No character mapping found for speaker '{speaker_id}' in dialogue line {i+1}.")
                
                # Update dialogue with all mapped characters
                if mapped_characters:
                    dialogue.characters = mapped_characters
                    dialogue.speaker = mapped_characters[0] if mapped_characters else None  # Keep first for backward compatibility
                    dialogue.resolution_method = "audio_propagation"
                    # Keep the LLM confidence if it was originally confident
                    if not dialogue.is_llm_confident:
                        dialogue.is_llm_confident = False  # Propagated assignments are not LLM confident
                    logger.debug(f"✅ Dialogue line {i+1}: mapped to characters {mapped_characters}")
                else:
                    logger.debug(f"Skipping dialogue line {i+1} as no character mapping found for any speakers: {dialogue.characters}")
            
            elif dialogue.speaker is None and not dialogue.characters:
                logger.debug(f"Skipping dialogue line {i+1} as no speakers assigned.")

        logger.info("Finished propagation of assignments.")
        return dialogue_lines
