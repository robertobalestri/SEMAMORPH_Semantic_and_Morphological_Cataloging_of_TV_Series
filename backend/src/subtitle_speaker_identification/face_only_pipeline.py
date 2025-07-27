"""
Face-only speaker identification pipeline.
Implements face-based speaker identification using existing clustering logic.
"""

import logging
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
            
            # Log statistics
            statistics = self._calculate_statistics(face_dialogues)
            self._log_statistics(statistics, "face_only")
            
            logger.info("✅ Face-only pipeline completed successfully")
            return face_dialogues
            
        except Exception as e:
            logger.error(f"❌ Face-only pipeline failed: {e}")
            return dialogue_lines 