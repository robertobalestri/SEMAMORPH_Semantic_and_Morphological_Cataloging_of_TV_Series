"""
Base classes for speaker identification pipelines.
Defines the abstract base class for all pipelines and common data structures.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
import os

from ..utils.logger_utils import setup_logging
from ..config import Config
from ..narrative_storage_management.narrative_models import DialogueLine # Import canonical DialogueLine

logger = setup_logging(__name__)

class SpeakerIdentificationConfig:
    """Configuration for speaker identification pipelines."""
    
    def __init__(self, config: Config):
        self._config = config

    def get_pipeline_mode(self) -> str:
        return self._config.speaker_identification_mode

    def is_audio_enabled(self) -> bool:
        return self._config.audio_enabled

    def is_face_enabled(self) -> bool:
        return self._config.face_enabled

    def is_character_mapping_enabled(self) -> bool:
        return self._config.character_mapping_enabled

    def get_auth_token(self) -> Optional[str]:
        return self._config.whisperx_auth_token

    # Diarization configuration properties
    @property
    def diarization_min_speakers(self) -> int:
        """Minimum number of speakers to detect in diarization."""
        return self._config.diarization_min_speakers
    
    @property
    def diarization_max_speakers(self) -> int:
        """Maximum number of speakers to detect in diarization."""
        return self._config.diarization_max_speakers
    
    @property
    def diarization_min_duration_on(self) -> float:
        """Minimum duration for a speaker turn in seconds."""
        return self._config.diarization_min_duration_on
    
    @property
    def diarization_min_duration_off(self) -> float:
        """Minimum duration of silence between speakers in seconds."""
        return self._config.diarization_min_duration_off
    
    @property
    def diarization_onset(self) -> float:
        """Onset threshold for speaker change detection (0.0-1.0)."""
        return self._config.diarization_onset
    
    @property
    def diarization_offset(self) -> float:
        """Offset threshold for speaker change detection (0.0-1.0)."""
        return self._config.diarization_offset
    
    @property
    def diarization_min_duration(self) -> float:
        """Minimum duration for any segment in seconds."""
        return self._config.diarization_min_duration
    
    @property
    def diarization_threshold(self) -> float:
        """Clustering threshold for diarization (0.0-1.0)."""
        return self._config.diarization_threshold

class BaseSpeakerIdentificationPipeline(ABC):
    """Abstract base class for speaker identification pipelines."""
    
    def __init__(self, path_handler, config: SpeakerIdentificationConfig, llm=None):
        self.path_handler = path_handler
        self.config = config
        self.llm = llm
        self.checkpoint_dir = self.path_handler.get_speaker_identification_checkpoint_dir()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @abstractmethod
    def run_pipeline(
        self,
        video_path: str,
        dialogue_lines: List[DialogueLine],
        episode_entities: List[Dict]
    ) -> List[DialogueLine]:
        """Run the speaker identification pipeline."""
        pass

    def _validate_dialogue_lines(self, dialogue_lines: List[DialogueLine]) -> bool:
        """Validate that dialogue lines are not empty."""
        if not dialogue_lines:
            logger.warning("⚠️ No dialogue lines provided to the pipeline")
            return False
        return True

    def _save_results(self, dialogue_lines: List[DialogueLine], pipeline_name: str):
        """Save pipeline results to a checkpoint file."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{pipeline_name}_results.json")
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump([d.to_dict() for d in dialogue_lines], f, indent=4)
            logger.info(f"📄 Saved {pipeline_name} results to {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save {pipeline_name} results: {e}")

    def _load_results(self, pipeline_name: str) -> Optional[List[DialogueLine]]:
        """Load pipeline results from a checkpoint file."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{pipeline_name}_results.json")
        logger.info(f"Attempting to load {pipeline_name} results from: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                
                logger.info(f"Successfully loaded raw data for {pipeline_name}. Data length: {len(data)}")

                dialogue_lines = []
                for d in data:
                    dialogue_lines.append(DialogueLine.from_dict(d)) # Use from_dict
                
                logger.info(f"📄 Loaded {pipeline_name} results from {checkpoint_path}. Total dialogue lines: {len(dialogue_lines)}")
                return dialogue_lines
            except Exception as e:
                logger.error(f"❌ Failed to load {pipeline_name} results: {e}")
        else:
            logger.info(f"Checkpoint file not found for {pipeline_name}: {checkpoint_path}")
        return None

    def _calculate_statistics(self, dialogue_lines: List[DialogueLine]) -> Dict:
        """Calculate statistics about the pipeline results."""
        total_lines = len(dialogue_lines)
        identified_lines = sum(1 for d in dialogue_lines if d.speaker)
        confident_lines = sum(1 for d in dialogue_lines if d.is_llm_confident)
        
        return {
            "total_lines": total_lines,
            "identified_lines": identified_lines,
            "confident_lines": confident_lines,
            "identification_rate": (identified_lines / total_lines * 100) if total_lines > 0 else 0,
            "confidence_rate": (confident_lines / total_lines * 100) if total_lines > 0 else 0,
        }

    def _log_statistics(self, statistics: Dict, pipeline_name: str):
        """Log pipeline statistics."""
        logger.info(f"📊 Statistics for {pipeline_name} pipeline:")
        logger.info(f"  - Total dialogue lines: {statistics['total_lines']}")
        logger.info(f"  - Identified speakers: {statistics['identified_lines']} ({statistics['identification_rate']:.2f}%)")
        logger.info(f"  - Confident identifications: {statistics['confident_lines']} ({statistics['confidence_rate']:.2f}%)")