"""
Complete speaker identification pipeline using audio + face + character mapping.
Implements hybrid speaker identification combining multiple modalities.
"""

import logging
from typing import List, Dict
import os

from .base_pipeline import BaseSpeakerIdentificationPipeline, DialogueLine, SpeakerIdentificationConfig
from .shared_components import SpeakerIdentificationComponents
from .audio_only_pipeline import AudioOnlyPipeline
from .face_only_pipeline import FaceOnlyPipeline
from ..utils.logger_utils import setup_logging
from ..path_handler import PathHandler
from ..utils.subtitle_utils import parse_srt_file

logger = setup_logging(__name__)

class CompletePipeline(BaseSpeakerIdentificationPipeline):
    """Complete speaker identification using audio + face + character mapping."""
    
    def __init__(self, path_handler, config: SpeakerIdentificationConfig, llm=None):
        super().__init__(path_handler, config, llm)
        self.components = SpeakerIdentificationComponents(path_handler, config, llm)
        self.audio_pipeline = AudioOnlyPipeline(path_handler, config, llm)
        self.face_pipeline = FaceOnlyPipeline(path_handler, config, llm)
    
    def run_pipeline(
        self,
        video_path: str,
        dialogue_lines: List[DialogueLine],
        episode_entities: List[Dict],
        plot_scenes: List[Dict]
    ) -> List[DialogueLine]:
        """Run complete speaker identification pipeline."""
        
        if not self._validate_dialogue_lines(dialogue_lines):
            return dialogue_lines
        
        logger.info("🎯 Running complete speaker identification pipeline")
        
        # Check if we can load from checkpoint
        checkpoint_dialogues = self._load_results("complete")
        if checkpoint_dialogues:
            logger.info("📄 Loaded complete pipeline results from checkpoint")
            statistics = self._calculate_statistics(checkpoint_dialogues)
            self._log_statistics(statistics, "complete")
            
            # Even when loading from checkpoint, we need to run the later steps if files don't exist
            # Step 7: Generate possible speakers SRT file
            possible_speakers_srt_path = self.path_handler.get_possible_speakers_srt_path()
            if not os.path.exists(possible_speakers_srt_path):
                logger.info("📝 Step 7: Generating possible speakers SRT file...")
                self._generate_possible_speakers_srt(checkpoint_dialogues)
            else:
                logger.info("📝 Step 7: Possible speakers SRT file already exists, skipping...")
            
            # Step 8: Generate plot from possible speakers SRT
            plot_possible_speakers_path = self.path_handler.get_plot_possible_speakers_path()
            if not os.path.exists(plot_possible_speakers_path):
                logger.info("📖 Step 8: Generating plot from possible speakers SRT...")
                self._generate_plot_from_possible_speakers()
            else:
                logger.info("📖 Step 8: Plot from possible speakers already exists, skipping...")
            
            # Step 9: Correct scene timestamps with new plot
            corrected_scene_timestamps_path = self.path_handler.get_corrected_scene_timestamps_path()
            if not os.path.exists(corrected_scene_timestamps_path):
                logger.info("🔄 Step 9: Correcting scene timestamps with new plot...")
                self._correct_scene_timestamps()
            else:
                logger.info("🔄 Step 9: Corrected scene timestamps already exist, skipping...")
            
            return checkpoint_dialogues
        
        try:
            # Step 1: Extract audio from video
            logger.info("🎵 Step 1: Extracting audio from video")
            audio_path = self.components.audio_processor.extract_audio_from_video(video_path)
            
            if not audio_path:
                logger.error("❌ Failed to extract audio from video")
                return dialogue_lines # Return current state if audio extraction fails
            
            # Step 2: Perform WhisperX alignment for precise timestamps
            logger.info("🗣️ Step 2: Performing WhisperX alignment for precise timestamps")
            srt_file_path = self.path_handler.get_srt_file_path()
            if not os.path.exists(srt_file_path):
                logger.error(f"❌ SRT file not found at {srt_file_path}")
                return dialogue_lines
            
            with open(srt_file_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            
            whisperx_aligned_result = self.components.audio_processor.align_subtitles_with_whisperx(audio_path, srt_content)
            
            if whisperx_aligned_result and "segments" in whisperx_aligned_result:
                logger.info("✅ WhisperX alignment complete. Updating dialogue lines with precise timestamps.")
                # Create a mapping from original dialogue text to WhisperX segments
                # This is a simplified mapping and might need more robust logic for complex cases
                whisperx_segments_map = {segment["text"].strip(): segment for segment in whisperx_aligned_result["segments"]}

                for dialogue in dialogue_lines:
                    # Find the corresponding WhisperX segment
                    # This assumes exact text match, which might not always be the case
                    matched_segment = whisperx_segments_map.get(dialogue.text.strip())
                    if matched_segment:
                        dialogue.start_time = matched_segment["start"]
                        dialogue.end_time = matched_segment["end"]
                        if "speaker" in matched_segment:
                            dialogue.speaker = matched_segment["speaker"]
            else:
                logger.warning("⚠️ WhisperX alignment failed or returned no segments. Proceeding with original timestamps.")

            # Step 3: Audio processing (now with potentially updated timestamps)
            logger.info("🎵 Step 3: Processing audio")
            audio_result = self._process_audio(video_path, dialogue_lines, episode_entities, plot_scenes)
            
            # Step 4: Face processing (now with potentially updated timestamps)
            logger.info("👤 Step 4: Processing faces")
            face_result = self._process_faces(video_path, dialogue_lines, episode_entities, plot_scenes)
            
            # Step 3: Combine audio and face results
            logger.info("🔄 Step 3: Combining modalities")
            combined_dialogues = self._combine_modalities(audio_result, face_result)
            
            # Step 4: Character mapping
            if self.config.is_character_mapping_enabled():
                logger.info("👥 Step 4: Mapping speakers to characters")
                mapped_dialogues = self.components.character_mapper.map_speakers_to_characters(
                    combined_dialogues, episode_entities
                )
            else:
                mapped_dialogues = combined_dialogues
            
            # Step 5: Calculate final confidence scores
            logger.info("📊 Step 5: Calculating final confidence scores")
            for dialogue in mapped_dialogues:
                dialogue.is_llm_confident = self.components.confidence_scorer.calculate_hybrid_confidence(dialogue)
                if dialogue.resolution_method == "audio_face_combined":
                    dialogue.resolution_method = "complete_pipeline"
            
            # Step 6: Save results
            self._save_results(mapped_dialogues, "complete")
            
            # Step 7: Generate possible speakers SRT file
            possible_speakers_srt_path = self.path_handler.get_possible_speakers_srt_path()
            if not os.path.exists(possible_speakers_srt_path):
                logger.info("📝 Step 7: Generating possible speakers SRT file...")
                self._generate_possible_speakers_srt(mapped_dialogues)
            else:
                logger.info("📝 Step 7: Possible speakers SRT file already exists, skipping...")
            
            # Step 8: Generate plot from possible speakers SRT
            plot_possible_speakers_path = self.path_handler.get_plot_possible_speakers_path()
            if not os.path.exists(plot_possible_speakers_path):
                logger.info("📖 Step 8: Generating plot from possible speakers SRT...")
                self._generate_plot_from_possible_speakers()
            else:
                logger.info("📖 Step 8: Plot from possible speakers already exists, skipping...")
            
            # Step 9: Correct scene timestamps with new plot
            corrected_scene_timestamps_path = self.path_handler.get_corrected_scene_timestamps_path()
            if not os.path.exists(corrected_scene_timestamps_path):
                logger.info("🔄 Step 9: Correcting scene timestamps with new plot...")
                self._correct_scene_timestamps()
            else:
                logger.info("🔄 Step 9: Corrected scene timestamps already exist, skipping...")
            
            # Log statistics
            statistics = self._calculate_statistics(mapped_dialogues)
            self._log_statistics(statistics, "complete")
            
            logger.info("✅ Complete pipeline completed successfully")
            return mapped_dialogues
            
        except Exception as e:
            logger.error(f"❌ Complete pipeline failed: {e}")
            return dialogue_lines
    
    def _process_audio(
        self,
        video_path: str,
        dialogue_lines: List[DialogueLine],
        episode_entities: List[Dict],
        plot_scenes: List[Dict]
    ) -> List[DialogueLine]:
        """Process audio using audio-only pipeline."""
        try:
            return self.audio_pipeline.run_pipeline(video_path, dialogue_lines, episode_entities, plot_scenes)
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            return dialogue_lines
    
    def _process_faces(
        self,
        video_path: str,
        dialogue_lines: List[DialogueLine],
        episode_entities: List[Dict],
        plot_scenes: List[Dict]
    ) -> List[DialogueLine]:
        """Process faces using face-only pipeline."""
        try:
            return self.face_pipeline.run_pipeline(video_path, dialogue_lines, episode_entities, plot_scenes)
        except Exception as e:
            logger.error(f"❌ Face processing failed: {e}")
            return dialogue_lines
    
    def _combine_modalities(
        self,
        audio_dialogues: List[DialogueLine],
        face_dialogues: List[DialogueLine]
    ) -> List[DialogueLine]:
        """Combine audio and face results using confidence-based decision."""
        
        combined_dialogues = []
        
        for i, (audio_dialogue, face_dialogue) in enumerate(zip(audio_dialogues, face_dialogues)):
            # Use the modality with higher confidence
            if audio_dialogue.is_llm_confident and not face_dialogue.is_llm_confident:
                combined_dialogue = audio_dialogue
                combined_dialogue.resolution_method = "audio_preferred"
            elif face_dialogue.is_llm_confident and not audio_dialogue.is_llm_confident:
                combined_dialogue = face_dialogue
                combined_dialogue.resolution_method = "face_preferred"
            elif audio_dialogue.is_llm_confident and face_dialogue.is_llm_confident:
                # Both confident - check if they agree
                if audio_dialogue.speaker == face_dialogue.speaker:
                    combined_dialogue = audio_dialogue
                    combined_dialogue.is_llm_confident = True
                    combined_dialogue.resolution_method = "audio_face_agreement"
                else:
                    # Disagreement - use audio (more reliable for speaker separation)
                    combined_dialogue = audio_dialogue
                    combined_dialogue.is_llm_confident = False
                    combined_dialogue.resolution_method = "audio_face_disagreement"
            else:
                # Neither confident - use audio as fallback
                combined_dialogue = audio_dialogue
                combined_dialogue.is_llm_confident = False
                combined_dialogue.resolution_method = "audio_fallback"
            
            combined_dialogues.append(combined_dialogue)
        
        return combined_dialogues
    
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