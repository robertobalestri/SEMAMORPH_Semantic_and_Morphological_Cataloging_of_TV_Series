"""
Character mapper for speaker identification.
Maps audio clusters to character names based on confident LLM assignments.
"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict

from ..utils.logger_utils import setup_logging
from ..narrative_storage_management.narrative_models import DialogueLine
from pyannote.audio import Pipeline
import numpy as np

logger = setup_logging(__name__)

class CharacterMapper:
    def __init__(self):
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", )

    def map_unassigned_speakers_with_embeddings(
        self, 
        dialogue_lines: List[DialogueLine], 
        speaker_mapping: Dict[str, str], 
        audio_path: str,
        embeddings=None,
        diarization=None
    ) -> Dict[str, str]:
        """Maps unassigned speakers using audio embeddings."""
        logger.info("🎙️ Mapping unassigned speakers with audio embeddings.")
        logger.info(f"🔊 Processing audio file: {audio_path}")
        logger.info(f"Received {len(dialogue_lines)} dialogue lines.")
        logger.info(f"Initial speaker mapping: {speaker_mapping}")

        # Use provided embeddings and diarization (diarization should only happen once in the pipeline)
        if embeddings is None or diarization is None:
            logger.error("❌ Embeddings and diarization must be provided - diarization should only happen once in the pipeline")
            return speaker_mapping
        
        logger.info("✅ Using provided embeddings and diarization.")

        # Build reference embeddings for known characters
        logger.info("Building reference embeddings for known characters...")
        reference_embeddings = defaultdict(list)
        speaker_labels = diarization.labels()
        if len(speaker_labels) == embeddings.shape[0]:
            for i, speaker_id in enumerate(speaker_labels):
                if speaker_id in speaker_mapping:
                    character_name = speaker_mapping[speaker_id]
                    reference_embeddings[character_name].append(embeddings[i])
        logger.info(f"✅ Built reference embeddings for {len(reference_embeddings)} characters.")

        # Calculate median embeddings for each character
        logger.info("Calculating median embeddings for each character...")
        character_median_embeddings = {
            character: np.median(embs, axis=0)
            for character, embs in reference_embeddings.items()
        }
        logger.info(f"✅ Calculated median embeddings for {len(character_median_embeddings)} characters.")

        # Identify unassigned speakers and assign them to the best match
        logger.info("Identifying and assigning unassigned speakers...")
        unassigned_speakers = [spk for spk in speaker_labels if spk not in speaker_mapping]
        logger.info(f"Found {len(unassigned_speakers)} unassigned speakers: {unassigned_speakers}")
        for i, speaker_id in enumerate(speaker_labels):
            if speaker_id in unassigned_speakers:
                logger.info(f"Processing unassigned speaker {speaker_id}...")
                embedding = embeddings[i]
                best_match = self._find_best_character_match(embedding, character_median_embeddings)
                if best_match:
                    logger.info(f"✅ Assigned unassigned speaker {speaker_id} to character {best_match} based on embedding similarity.")
                    speaker_mapping[speaker_id] = best_match
        
        logger.info("✅ Finished mapping unassigned speakers.")
        return speaker_mapping

    def _find_best_character_match(self, embedding: np.ndarray, character_embeddings: Dict[str, np.ndarray]) -> Optional[str]:
        """
        Finds the best character match for a given embedding.
        """
        best_match = None
        max_similarity = -1

        for character, char_embedding in character_embeddings.items():
            similarity = self._cosine_similarity(embedding, char_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = character

        return best_match

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two embeddings.
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
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
                if hasattr(dialogue, 'original_llm_speaker') and dialogue.original_llm_speaker and dialogue.original_llm_is_confident:
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
                # IMPORTANT: If the LLM was confident about this dialogue, preserve that assignment
                if dialogue.is_llm_confident and dialogue.original_llm_speaker and dialogue.original_llm_is_confident:
                    logger.debug(f"🔒 Preserving confident LLM assignment for dialogue {i+1}: {dialogue.speaker}")
                    # Use the validated speaker name (dialogue.speaker) instead of the raw original_llm_speaker
                    # The speaker field already contains the validated character name from the speaker mapping
                    
                    # BUT: We still need to update the characters array with the mapped character names
                    mapped_characters = []
                    for speaker_id in dialogue.characters:
                        if speaker_id in speaker_mapping:
                            character_name = speaker_mapping[speaker_id]
                            mapped_characters.append(character_name)
                            logger.debug(f"🔒 Mapped confident speaker '{speaker_id}' to character '{character_name}' for dialogue line {i+1}.")
                        else:
                            logger.debug(f"🔒 No mapping found for confident speaker '{speaker_id}' in dialogue line {i+1}.")
                    
                    # Update the characters array with mapped names
                    if mapped_characters:
                        dialogue.characters = mapped_characters
                        logger.debug(f"🔒 Updated characters array for confident dialogue {i+1}: {mapped_characters}")
                    
                    dialogue.resolution_method = "llm_confident_preserved"
                    continue
                
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
                    # Update speaker field to match characters
                    if len(mapped_characters) == 1:
                        dialogue.speaker = mapped_characters[0]
                    else:
                        dialogue.speaker = " OR ".join(mapped_characters)
                    dialogue.resolution_method = "audio_propagation"
                    # Keep the LLM confidence if it was originally confident
                    if not dialogue.is_llm_confident:
                        dialogue.is_llm_confident = False  # Propagated assignments are not LLM confident
                    logger.debug(f"✅ Dialogue line {i+1}: mapped to characters {mapped_characters}, speaker: {dialogue.speaker}")
                else:
                    logger.debug(f"Skipping dialogue line {i+1} as no character mapping found for any speakers: {dialogue.characters}")
            
            elif dialogue.speaker is None and not dialogue.characters:
                logger.debug(f"Skipping dialogue line {i+1} as no speakers assigned.")

        logger.info("Finished propagation of assignments.")
        return dialogue_lines
