"""
Audio-only speaker identification pipeline using Pyannote.
Implements audio-based speaker diarization and timeline alignment.
"""

import logging
from typing import List, Dict
import os

from .base_pipeline import BaseSpeakerIdentificationPipeline, SpeakerIdentificationConfig
from ..narrative_storage_management.narrative_models import DialogueLine
from .shared_components import SpeakerIdentificationComponents
from ..utils.logger_utils import setup_logging
from ..path_handler import PathHandler
from .speaker_identifier import SpeakerIdentifier
import json

logger = setup_logging(__name__)

class AudioOnlyPipeline(BaseSpeakerIdentificationPipeline):
    """Audio-only speaker identification using Pyannote."""
    
    def __init__(self, path_handler, config: SpeakerIdentificationConfig, llm=None):
        super().__init__(path_handler, config, llm)
        self.components = SpeakerIdentificationComponents(path_handler, config, llm)
        self.llm_speaker_identifier = SpeakerIdentifier(llm, path_handler.get_series()) # Instantiate SpeakerIdentifier
    
    def run_pipeline(
        self,
        video_path: str,
        dialogue_lines: List[DialogueLine],
        episode_entities: List[Dict],
        plot_scenes: List[Dict] # New parameter
    ) -> List[DialogueLine]:
        """Run audio-only speaker identification."""
        
        logger.info(f"🎵 Starting audio-only pipeline for {self.path_handler.get_episode_code()}")
        logger.info(f"📋 Parameters: video_path={video_path}, dialogue_lines_count={len(dialogue_lines)}, episode_entities_count={len(episode_entities)}, plot_scenes_count={len(plot_scenes)}")
        
        if not self._validate_dialogue_lines(dialogue_lines):
            logger.warning("⚠️ Dialogue lines validation failed")
            return dialogue_lines
        
        logger.info("🎵 Running audio-only speaker identification")
        
        # Check if we can load from checkpoint
        logger.info("📂 Checking for existing checkpoint...")
        checkpoint_dialogues = self._load_results("audio_only")
        if checkpoint_dialogues:
            logger.info("📄 Loaded audio-only results from checkpoint")
            statistics = self._calculate_statistics(checkpoint_dialogues)
            self._log_statistics(statistics, "audio_only")
            return checkpoint_dialogues
        
        try:
            # Step 1: Initial LLM pass for confidence tagging
            logger.info("🧠 Step 1: Initial LLM pass for confidence tagging...")
            llm_checkpoint_dialogues = self._load_results("llm_initial_pass")
            if llm_checkpoint_dialogues:
                logger.info("📄 Loaded LLM initial pass results from checkpoint")
                dialogue_lines_with_confidence = llm_checkpoint_dialogues
            else:
                logger.info("🧠 Running initial LLM pass for confidence tagging...")
                dialogue_lines_with_confidence = self.llm_speaker_identifier.identify_speakers_for_episode(
                    plot_scenes, # Pass plot_scenes here
                    dialogue_lines,
                    episode_entities=episode_entities # Pass episode_entities
                )
                logger.info(f"✅ LLM pass completed. Dialogue lines with confidence: {len(dialogue_lines_with_confidence)}")
                self._save_results(dialogue_lines_with_confidence, "llm_initial_pass")
            
            # Step 2: Get audio path
            logger.info("🎵 Step 2: Getting audio path...")
            audio_path = self.path_handler.get_audio_file_path()
            logger.info(f"🎵 Audio path: {audio_path}")
            
            # Step 3: Check if audio path exists
            if not audio_path:
                logger.error("❌ Failed to extract audio from video")
                return dialogue_lines_with_confidence # Return current state if audio extraction fails
            
            # Step 3: Perform speaker diarization only (transcription and alignment already done during initial step)
            logger.info("🗣️ Step 3: Running speaker diarization...")
            srt_file_path = self.path_handler.get_srt_file_path()
            logger.info(f"📝 SRT file path: {srt_file_path}")
            if not os.path.exists(srt_file_path):
                logger.error(f"❌ SRT file not found at {srt_file_path}")
                return dialogue_lines_with_confidence
            
            logger.info("🎤 About to call diarize_speakers_with_pyannote...")
            whisperx_diarization_result = self.components.audio_processor.diarize_speakers_with_pyannote(
                audio_path, srt_file_path
            )
            logger.info(f"✅ Diarization completed. Result type: {type(whisperx_diarization_result)}")
            logger.info(f"✅ Diarization result keys: {list(whisperx_diarization_result.keys()) if isinstance(whisperx_diarization_result, dict) else 'Not a dict'}")
            
            if whisperx_diarization_result and "segments" in whisperx_diarization_result:
                logger.info("✅ Speaker diarization complete. Updating dialogue lines with speaker labels.")
                
                # The audio processor now returns segments with direct speaker assignments
                # No need for timeline alignment since speakers are already assigned
                whisperx_segments = whisperx_diarization_result["segments"]
                logger.info(f"📊 WhisperX segments count: {len(whisperx_segments)}")
                
                # Update dialogue lines with speaker assignments from diarization
                logger.info("🔄 Updating dialogue lines with speaker assignments...")
                dialogue_lines_with_audio_clusters = self._update_dialogue_with_speaker_assignments(
                    dialogue_lines_with_confidence,
                    whisperx_segments
                )
                logger.info(f"✅ Updated dialogue lines count: {len(dialogue_lines_with_audio_clusters)}")
            else:
                logger.warning("⚠️ Speaker diarization failed or returned no segments. Proceeding with original timestamps and no speaker assignments from diarization.")
                dialogue_lines_with_audio_clusters = dialogue_lines_with_confidence # Proceed with original if no results
            
            # Step 5: Map audio speakers to confident character names
            logger.info("👥 Step 5: Mapping audio speakers to confident character names...")
            audio_speaker_to_character_mapping = self.components.character_mapper.map_audio_clusters_to_characters(
                dialogue_lines_with_audio_clusters
            )
            logger.info(f"✅ Audio speaker to character mapping: {audio_speaker_to_character_mapping}")

            # Step 5.5: Save the audio speaker to character mapping
            logger.info("💾 Step 5.5: Saving audio speaker to character mapping...")
            mapping_path = os.path.join(self.checkpoint_dir, "audio_speaker_to_character_mapping.json")
            try:
                with open(mapping_path, 'w') as f:
                    json.dump(audio_speaker_to_character_mapping, f, indent=4)
                logger.info(f"📄 Saved audio speaker to character mapping to {mapping_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save audio speaker to character mapping: {e}")

            # Step 6: Propagate confident assignments to all dialogues with the same speaker
            logger.info("🔄 Step 6: Propagating confident assignments...")
            final_dialogues = self.components.character_mapper.propagate_assignments(
                dialogue_lines_with_audio_clusters,
                audio_speaker_to_character_mapping
            )
            logger.info(f"✅ Final dialogues count: {len(final_dialogues)}")
            
            # Step 7: Save results
            logger.info("💾 Step 7: Saving results...")
            self._save_results(final_dialogues, "audio_only")
            
            # Log statistics
            logger.info("📊 Calculating and logging statistics...")
            statistics = self._calculate_statistics(final_dialogues)
            self._log_statistics(statistics, "audio_only")
            
            logger.info("✅ Audio-only pipeline completed successfully")
            return final_dialogues
            
        except Exception as e:
            logger.error(f"❌ Audio-only pipeline failed: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            logger.error(f"❌ Exception args: {e.args}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            # In case of failure, return the dialogue lines as they were before the pipeline run
            return dialogue_lines
    
    def _validate_audio_components(self) -> bool:
        """Validate that audio components are properly initialized."""
        if not self.components.audio_processor.pyannote_pipeline:
            logger.error("❌ Pyannote pipeline not initialized")
            return False
        
        auth_token = self.config.get_auth_token()
        if not auth_token:
            logger.error("❌ No HuggingFace auth token provided")
            return False
        
        return True
    
    def _update_dialogue_with_speaker_assignments(
        self, 
        dialogue_lines: List[DialogueLine], 
        whisperx_segments: List[Dict]
    ) -> List[DialogueLine]:
        """
        Update dialogue lines with speaker assignments from diarization.
        
        Args:
            dialogue_lines: List of dialogue lines with LLM confidence
            whisperx_segments: List of segments with speaker assignments from diarization
            
        Returns:
            Updated dialogue lines with speaker assignments
        """
        logger.info(f"🎯 Updating {len(dialogue_lines)} dialogue lines with speaker assignments from {len(whisperx_segments)} segments")
        
        # Create a mapping from segment text to speaker assignments (all speakers equal)
        segment_speaker_map = {}
        for segment in whisperx_segments:
            if "text" in segment:
                text_key = segment["text"].strip()
                if "all_speakers" in segment:
                    # New format with multiple speakers (all equal)
                    segment_speaker_map[text_key] = {
                        'all_speakers': segment.get('all_speakers', []),
                        'speakers': segment.get('speakers', [])
                    }
                elif "speaker" in segment:
                    # Legacy format with single speaker
                    segment_speaker_map[text_key] = {
                        'all_speakers': [segment["speaker"]],
                        'speakers': [{'speaker': segment["speaker"], 'overlap_duration': 1.0, 'overlap_percentage': 100.0}]
                    }
        
        # Update dialogue lines with speaker assignments
        updated_dialogue_lines = []
        for dialogue in dialogue_lines:
            # Store original LLM speaker assignment before overwriting
            if dialogue.is_llm_confident and dialogue.speaker:
                dialogue.original_llm_speaker = dialogue.speaker
                dialogue.original_llm_is_confident = dialogue.is_llm_confident
                logger.debug(f"💾 Dialogue {dialogue.index}: stored original LLM speaker '{dialogue.speaker}'")
            
            # Try to find matching segment by text
            dialogue_text = dialogue.text.strip()
            speaker_info = segment_speaker_map.get(dialogue_text)
            
            if speaker_info:
                # Update the dialogue line with all speaker assignments from diarization (all equal)
                dialogue.characters = speaker_info['all_speakers']  # Store all characters
                dialogue.speaker = speaker_info['all_speakers'][0] if speaker_info['all_speakers'] else None  # Keep first for backward compatibility
                dialogue.resolution_method = "audio_diarization"
                
                # Store additional speaker information for potential use
                dialogue.audio_cluster_assignments = speaker_info['speakers']
                dialogue.candidate_speakers = speaker_info['all_speakers']
                
                logger.debug(f"🎯 Dialogue {dialogue.index}: assigned {len(speaker_info['all_speakers'])} speakers via diarization: {speaker_info['all_speakers']}")
                if len(speaker_info['all_speakers']) > 1:
                    logger.debug(f"   📊 Multiple speakers detected: {speaker_info['all_speakers']}")
            else:
                # Keep original speaker assignment if no match found
                logger.debug(f"⚠️ Dialogue {dialogue.index}: no speaker match found in diarization segments")
            
            updated_dialogue_lines.append(dialogue)
        
        # Count assignments and multiple character cases
        assigned_count = sum(1 for d in updated_dialogue_lines if d.characters and d.resolution_method == "audio_diarization")
        multiple_character_count = sum(1 for d in updated_dialogue_lines if d.characters and len(d.characters) > 1)
        
        logger.info(f"✅ Updated {assigned_count}/{len(updated_dialogue_lines)} dialogue lines with character assignments from diarization")
        if multiple_character_count > 0:
            logger.info(f"📊 Found {multiple_character_count} dialogue lines with multiple characters: {multiple_character_count}")
        
        return updated_dialogue_lines