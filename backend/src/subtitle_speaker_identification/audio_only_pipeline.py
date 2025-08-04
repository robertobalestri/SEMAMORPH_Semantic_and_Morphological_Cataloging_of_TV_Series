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
import traceback

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
        plot_scenes: List[Dict]
    ) -> List[DialogueLine]:
        """Run audio-only speaker identification in a linear flow."""
        
        logger.info(f"🎵 Starting audio-only pipeline for {self.path_handler.get_episode_code()}")
        logger.info(f"📋 Parameters: video_path={video_path}, dialogue_lines_count={len(dialogue_lines)}, episode_entities_count={len(episode_entities)}, plot_scenes_count={len(plot_scenes)}")
        
        if not self._validate_dialogue_lines(dialogue_lines):
            logger.warning("⚠️ Dialogue lines validation failed")
            return dialogue_lines
        
        try:
            # Step 0: Attempt to load final results from checkpoint
            logger.info("📂 Checking for existing audio_only checkpoint...")
            final_dialogues = self._load_results("audio_only")
            
            if final_dialogues:
                logger.info("📄 Loaded audio-only results directly from checkpoint. Skipping main pipeline.")
            else:
                # === Main Pipeline Logic (Steps 1-8) ===
                logger.info("✅ No final checkpoint found. Running full audio-only pipeline...")

                # Step 1: Initial LLM pass for confidence tagging
                logger.info("🧠 Step 1: Initial LLM pass for confidence tagging...")
                llm_checkpoint_dialogues = self._load_results("llm_initial_pass")
                if llm_checkpoint_dialogues:
                    logger.info("📄 Loaded LLM initial pass results from checkpoint")
                    dialogue_lines_with_confidence = llm_checkpoint_dialogues
                else:
                    logger.info("🧠 Running initial LLM pass for confidence tagging...")
                    dialogue_lines_with_confidence = self.llm_speaker_identifier.identify_speakers_for_episode(
                        plot_scenes,
                        dialogue_lines,
                        episode_entities=episode_entities
                    )
                    logger.info(f"✅ LLM pass completed. Dialogue lines with confidence: {len(dialogue_lines_with_confidence)}")
                    self._save_results(dialogue_lines_with_confidence, "llm_initial_pass")
                
                # Step 2: Get audio path
                logger.info("🎵 Step 2: Getting audio path...")
                audio_path = self.path_handler.get_audio_file_path()
                logger.info(f"🎵 Audio path: {audio_path}")
                if not audio_path:
                    logger.error("❌ Failed to extract audio from video")
                    return dialogue_lines_with_confidence
                
                # Step 3: Perform speaker diarization
                logger.info("🗣️ Step 3: Running speaker diarization...")
                srt_file_path = self.path_handler.get_srt_file_path()
                if not os.path.exists(srt_file_path):
                    logger.error(f"❌ SRT file not found at {srt_file_path}")
                    return dialogue_lines_with_confidence
                
                whisperx_diarization_result = self.components.audio_processor.diarize_speakers_with_pyannote(
                    audio_path, srt_file_path
                )
                
                if whisperx_diarization_result and "segments" in whisperx_diarization_result:
                    logger.info("✅ Speaker diarization complete. Updating dialogue lines.")
                    whisperx_segments = whisperx_diarization_result["segments"]
                    embeddings = whisperx_diarization_result.get("embeddings")
                    diarization = whisperx_diarization_result.get("diarization")
                    
                    dialogue_lines_with_audio_clusters = self._update_dialogue_with_speaker_assignments(
                        dialogue_lines_with_confidence,
                        whisperx_segments
                    )
                else:
                    logger.warning("⚠️ Speaker diarization failed or returned no segments. Proceeding without speaker assignments from diarization.")
                    dialogue_lines_with_audio_clusters = dialogue_lines_with_confidence
                
                # Step 5: Map audio speakers to confident character names
                logger.info("👥 Step 5: Mapping audio speakers to confident character names...")
                audio_speaker_to_character_mapping = self.components.character_mapper.map_audio_clusters_to_characters(
                    dialogue_lines_with_audio_clusters
                )
                logger.info(f"✅ Audio speaker to character mapping: {audio_speaker_to_character_mapping}")

                # Step 5.5: Save the mapping
                logger.info("💾 Step 5.5: Saving audio speaker to character mapping...")
                mapping_path = os.path.join(self.checkpoint_dir, "audio_speaker_to_character_mapping.json")
                try:
                    with open(mapping_path, 'w') as f:
                        json.dump(audio_speaker_to_character_mapping, f, indent=4)
                    logger.info(f"📄 Saved audio speaker to character mapping to {mapping_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save audio speaker to character mapping: {e}")

                # Step 6: Map unassigned speakers with embeddings
                logger.info("🎙️ Step 6: Mapping unassigned speakers with embeddings...")
                audio_speaker_to_character_mapping = self.components.character_mapper.map_unassigned_speakers_with_embeddings(
                    dialogue_lines_with_audio_clusters,
                    audio_speaker_to_character_mapping,
                    audio_path,
                    embeddings,
                    diarization
                )

                # Step 7: Propagate confident assignments
                logger.info("🔄 Step 7: Propagating confident assignments...")
                final_dialogues = self.components.character_mapper.propagate_assignments(
                    dialogue_lines_with_audio_clusters,
                    audio_speaker_to_character_mapping
                )
                
                # Step 8: Save final results
                logger.info("💾 Step 8: Saving final audio_only results...")
                self._save_results(final_dialogues, "audio_only")
            
            # === Consolidated Post-Processing (Steps 9-11) ===
            # This block now runs linearly after the main results are available,
            # either from the checkpoint or from the full run above.
            
            logger.info("🚀 Running post-processing steps (9-11)...")

            # Step 9: Generate possible speakers SRT file
            possible_speakers_srt_path = self.path_handler.get_possible_speakers_srt_path()
            if not os.path.exists(possible_speakers_srt_path):
                logger.info("📝 Step 9: Generating possible speakers SRT file...")
                self._generate_possible_speakers_srt(final_dialogues)
            else:
                logger.info("📝 Step 9: Possible speakers SRT file already exists, skipping...")

            # Step 10: Generate plot from possible speakers SRT
            plot_possible_speakers_path = self.path_handler.get_plot_possible_speakers_path()
            if not os.path.exists(plot_possible_speakers_path):
                logger.info("📖 Step 10: Generating plot from possible speakers SRT...")
                self._generate_plot_from_possible_speakers()
            else:
                logger.info("📖 Step 10: Plot from possible speakers already exists, skipping...")

            # Step 11: Correct scene timestamps with new plot
            corrected_scene_timestamps_path = self.path_handler.get_corrected_scene_timestamps_path()
            if not os.path.exists(corrected_scene_timestamps_path):
                logger.info("🔄 Step 11: Correcting scene timestamps with new plot...")
                self._correct_scene_timestamps()
            else:
                logger.info("🔄 Step 11: Corrected scene timestamps already exist, skipping...")
            
            # Final statistics and return
            logger.info("📊 Calculating and logging final statistics...")
            statistics = self._calculate_statistics(final_dialogues)
            self._log_statistics(statistics, "audio_only")
            
            logger.info("✅ Audio-only pipeline completed successfully")
            return final_dialogues
            
        except Exception as e:
            logger.error(f"❌ Audio-only pipeline failed: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            logger.error(f"❌ Exception args: {e.args}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
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
                dialogue.resolution_method = "audio_diarization"
                
                # Only update speaker field if LLM was NOT confident
                if not dialogue.is_llm_confident:
                    # Use all speakers, not just the first one
                    if speaker_info['all_speakers']:
                        if len(speaker_info['all_speakers']) == 1:
                            dialogue.speaker = speaker_info['all_speakers'][0]
                        else:
                            # For multiple speakers, join them with " & " or similar
                            dialogue.speaker = " OR ".join(speaker_info['all_speakers'])
                        logger.debug(f"🎯 Dialogue {dialogue.index}: updated speaker to '{dialogue.speaker}' (LLM was not confident)")
                    else:
                        dialogue.speaker = None
                else:
                    logger.debug(f"🎯 Dialogue {dialogue.index}: kept original speaker '{dialogue.speaker}' (LLM was confident)")
                
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
    
    def _generate_possible_speakers_srt(self, dialogue_lines: List[DialogueLine]) -> None:
        """
        Generate an SRT file with possible speakers based on the logic:
        - if is_self_presentation: ALWAYS the original LLM speaker
        - if is_llm_confident: ALWAYS the original LLM speaker  
        - if not is_llm_confident: the original LLM speaker then a slash "/" then the value of speaker with in parenthesis (PVM) (to indicate that can be the original or the audio match)
        
        Args:
            dialogue_lines: List of dialogue lines with speaker assignments
        """
        logger.info("📝 Generating possible speakers SRT file...")
        
        try:
            # Get the output path
            output_path = self.path_handler.get_possible_speakers_srt_path()
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, dialogue in enumerate(dialogue_lines, 1):
                    # Format timestamp
                    start_time = self._format_timestamp(dialogue.start_time)
                    end_time = self._format_timestamp(dialogue.end_time)
                    
                    # Determine speaker label based on the logic
                    if dialogue.is_self_presentation:
                        # ALWAYS the original LLM speaker
                        speaker_label = dialogue.original_llm_speaker or dialogue.speaker or "Unknown"
                    elif dialogue.is_llm_confident:
                        # ALWAYS the original LLM speaker
                        speaker_label = dialogue.original_llm_speaker or dialogue.speaker or "Unknown"
                    else:
                        # Original LLM speaker / Audio match (PVM)
                        original_speaker = dialogue.original_llm_speaker or dialogue.speaker or "Unknown"
                        audio_speaker = dialogue.speaker or "Unknown"
                        speaker_label = f"{original_speaker} / {audio_speaker} (PVM)"
                    
                    # Write SRT entry
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{speaker_label}: {dialogue.text}\n\n")
            
            logger.info(f"✅ Generated possible speakers SRT file: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate possible speakers SRT file: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _generate_plot_from_possible_speakers(self) -> None:
        """Generate plot from possible speakers SRT file."""
        try:
            from .plot_generator import PlotGenerator
            plot_generator = PlotGenerator(self.path_handler, self.llm)
            plot_generator.generate_plot_from_possible_speakers()
        except Exception as e:
            logger.error(f"❌ Failed to generate plot from possible speakers: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
    
    def _correct_scene_timestamps(self) -> None:
        """Correct scene timestamps with new plot."""
        try:
            from .plot_generator import PlotGenerator
            plot_generator = PlotGenerator(self.path_handler, self.llm)
            
            # Get the plot path
            plot_path = self.path_handler.get_plot_possible_speakers_path()
            if os.path.exists(plot_path):
                plot_generator.correct_scene_timestamps(plot_path)
            else:
                logger.warning(f"⚠️ Plot file not found: {plot_path}")
        except Exception as e:
            logger.error(f"❌ Failed to correct scene timestamps: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")