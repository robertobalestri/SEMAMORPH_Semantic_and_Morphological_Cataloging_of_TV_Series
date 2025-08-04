"""
Face-only speaker identification pipeline.
Implements face-based speaker identification using existing clustering logic.
"""

import logging
import os
from typing import List, Dict

from .base_pipeline import BaseSpeakerIdentificationPipeline, DialogueLine, SpeakerIdentificationConfig
from .shared_components import SpeakerIdentificationComponents
from ..utils.logger_utils import setup_logging
from .speaker_identifier import SpeakerIdentifier # New import

logger = setup_logging(__name__)

class FaceOnlyPipeline(BaseSpeakerIdentificationPipeline):
    """Face-only speaker identification using face clustering."""
    
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
        """Run face-only speaker identification."""
        
        if not self._validate_dialogue_lines(dialogue_lines):
            return dialogue_lines
        
        logger.info("👤 Running face-only speaker identification")
        
        # Check if we can load from checkpoint
        checkpoint_dialogues = self._load_results("face_only")
        if checkpoint_dialogues:
            logger.info("📄 Loaded face-only results from checkpoint")
            statistics = self._calculate_statistics(checkpoint_dialogues)
            self._log_statistics(statistics, "face_only")
            
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
            # Step 1: Initial LLM pass for confidence tagging
            llm_checkpoint_dialogues = self._load_results("llm_initial_pass")
            if llm_checkpoint_dialogues:
                logger.info("📄 Loaded LLM initial pass results from checkpoint")
                dialogue_lines_with_confidence = llm_checkpoint_dialogues
            else:
                logger.info("🧠 Step 1: Running initial LLM pass for confidence tagging")
                dialogue_lines_with_confidence = self.llm_speaker_identifier.identify_speakers_for_episode(
                    plot_scenes,
                    dialogue_lines,
                    episode_entities=episode_entities
                )
                self._save_results(dialogue_lines_with_confidence, "llm_initial_pass")
            
            # Step 2: Extract faces from video
            logger.info("👤 Step 2: Extracting faces from video")
            face_data = self.components.face_processor.extract_faces_from_video(
                video_path, dialogue_lines_with_confidence
            )
            
            if not face_data['faces']:
                logger.warning("⚠️ No faces detected in video")
                return dialogue_lines
            
            # Step 2: Generate face embeddings
            logger.info("🔢 Step 2: Generating face embeddings")
            embeddings = self.components.face_processor.generate_embeddings(face_data)
            
            if not embeddings:
                logger.warning("⚠️ No face embeddings generated")
                return dialogue_lines
            
            # Step 3: Cluster faces
            logger.info("🎯 Step 3: Clustering faces")
            clusters = self.components.face_processor.cluster_faces(embeddings)
            
            if not clusters['clusters']:
                logger.warning("⚠️ No face clusters created")
                return dialogue_lines
            
            # Step 4: Assign speakers based on face clusters
            logger.info("👥 Step 4: Assigning speakers from face clusters")
            face_dialogues = self.components.face_processor.assign_speakers_from_clusters(
                dialogue_lines, clusters
            )
            
            # Step 5: Calculate confidence scores
            logger.info("📊 Step 5: Calculating confidence scores")
            for dialogue in face_dialogues:
                dialogue.is_llm_confident = self.components.confidence_scorer.calculate_face_confidence(dialogue)
                dialogue.resolution_method = "face_only"
            
            # Step 6: Save results
            self._save_results(face_dialogues, "face_only")
            
            # Step 7: Generate possible speakers SRT file
            possible_speakers_srt_path = self.path_handler.get_possible_speakers_srt_path()
            if not os.path.exists(possible_speakers_srt_path):
                logger.info("📝 Step 7: Generating possible speakers SRT file...")
                self._generate_possible_speakers_srt(face_dialogues)
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
            statistics = self._calculate_statistics(face_dialogues)
            self._log_statistics(statistics, "face_only")
            
            logger.info("✅ Face-only pipeline completed successfully")
            return face_dialogues
            
        except Exception as e:
            logger.error(f"❌ Face-only pipeline failed: {e}")
            return dialogue_lines
    
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