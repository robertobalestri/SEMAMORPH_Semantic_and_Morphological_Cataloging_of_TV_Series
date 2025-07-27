"""
Audio processing utilities for speaker identification.
Handles audio extraction, WhisperX alignment, and speaker diarization.
"""

import os
import gc
import logging
import torch
import srt
import pandas as pd
import subprocess
import json
from typing import Dict, List, Any

from ..utils.logger_utils import setup_logging
from ..path_handler import PathHandler

logger = setup_logging(__name__)


class AudioProcessor:
    """
    Handles audio processing tasks including transcription, alignment, 
    and speaker diarization with tunable hyperparameters.
    """

    def __init__(self, config, path_handler: PathHandler):
        """
        Initializes the AudioProcessor.

        Args:
            config: A configuration object with diarization parameters and a `get_auth_token` method.
            path_handler: An instance of PathHandler for managing file paths.
        """
        self.config = config
        self.path_handler = path_handler
        self.pyannote_pipeline = None
        self._init_pyannote_pipeline()

    def _init_pyannote_pipeline(self):
        """
        Loads the Pyannote pipeline and tunes its hyperparameters based on the config.
        This follows the official recommended practice for adapting a pre-trained pipeline.
        """
        try:
            # Handle both Config and SpeakerIdentificationConfig objects
            if hasattr(self.config, 'get_auth_token'):
                auth_token = self.config.get_auth_token()
            elif hasattr(self.config, 'whisperx_auth_token'):
                auth_token = self.config.whisperx_auth_token
            else:
                auth_token = None
                
            if not auth_token:
                logger.warning("⚠️ No HuggingFace auth token provided. Pyannote will be unavailable.")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Initializing Pyannote pipeline on {device}")

            from pyannote.audio import Pipeline
            self.pyannote_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )

            # 1. Get the default hyperparameters from the loaded pipeline
            default_params = self.pyannote_pipeline.parameters(instantiated=True)
            logger.info(f"📋 Default pipeline hyperparameters: {default_params}")

            # 2. Create a dictionary of parameters to update, pulling from our config.
            #    This is robust because it only overrides the parameters we explicitly provide.
            updated_params = {}
            
            # The current Pyannote pipeline version uses a different parameter structure
            # Based on the default_params, we need to use 'segmentation' and 'clustering' sections
            
            # Handle segmentation parameters
            if 'segmentation' in default_params:
                updated_params["segmentation"] = default_params["segmentation"].copy()
                # Update min_duration_off if it exists in segmentation
                if 'min_duration_off' in default_params['segmentation']:
                    updated_params["segmentation"]["min_duration_off"] = getattr(
                        self.config, 'diarization_min_duration_off', 
                        default_params['segmentation']['min_duration_off']
                    )
                else:
                    updated_params["segmentation"]["min_duration_off"] = getattr(
                        self.config, 'diarization_min_duration_off', 0.05
                    )
            else:
                # Create segmentation section if it doesn't exist
                updated_params["segmentation"] = {
                    "min_duration_off": getattr(self.config, 'diarization_min_duration_off', 0.05)
                }
            
            # Handle clustering parameters
            if 'clustering' in default_params:
                updated_params["clustering"] = default_params["clustering"].copy()
                # Update threshold if it exists in clustering
                if 'threshold' in default_params['clustering']:
                    updated_params["clustering"]["threshold"] = getattr(
                        self.config, 'diarization_threshold', 
                        default_params['clustering']['threshold']
                    )
                else:
                    updated_params["clustering"]["threshold"] = getattr(
                        self.config, 'diarization_threshold', 0.4
                    )
                
                # Update min_cluster_size to be more permissive for TV shows
                # Default is 12, but for TV shows with many short dialogues, we need lower values
                if 'min_cluster_size' in default_params['clustering']:
                    # Use a lower min_cluster_size for better speaker detection in TV shows
                    updated_params["clustering"]["min_cluster_size"] = 3  # Much more permissive than default 12
                else:
                    updated_params["clustering"]["min_cluster_size"] = 3
            else:
                # Create clustering section if it doesn't exist
                updated_params["clustering"] = {
                    "threshold": getattr(self.config, 'diarization_threshold', 0.4),
                    "min_cluster_size": 3  # More permissive for TV shows
                }
            
            logger.info(f"🔧 Overriding pipeline defaults with: {updated_params}")

            # 3. Instantiate the pipeline with our modified parameters
            self.pyannote_pipeline.instantiate(updated_params)
            self.pyannote_pipeline.to(torch.device(device))

            logger.info("✅ Pyannote diarization pipeline initialized and tuned successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to initialize and tune Pyannote pipeline: {e}", exc_info=True)
            self.pyannote_pipeline = None

    def diarize_speakers_with_pyannote(self, audio_path: str, srt_path: str) -> Dict | None:
        """
        Performs speaker diarization using the customized pipeline by assigning speakers 
        to existing aligned SRT segments.

        Args:
            audio_path: Path to the audio file (must be 16kHz mono WAV).
            srt_path: Path to the SRT subtitle file (already aligned).

        Returns:
            A dictionary with a "segments" key containing a list of segments with 
            speaker assignments, or None if the process fails.
        """
        if not self.pyannote_pipeline:
            logger.error("❌ Pyannote diarization model not loaded. Cannot perform diarization.")
            return None

        try:
            # 1. Load existing SRT segments
            if not os.path.exists(srt_path):
                logger.error(f"❌ SRT file not found at {srt_path}")
                return None
            
            logger.info(f"Loading SRT from {srt_path}")
            with open(srt_path, 'r', encoding='utf-8') as f:
                loaded_subs = list(srt.parse(f.read()))
            
            segments = [{
                "start": sub.start.total_seconds(),
                "end": sub.end.total_seconds(),
                "text": sub.content
            } for sub in loaded_subs]
            logger.info(f"✅ Loaded {len(segments)} segments from SRT file.")

            # 2. Define runtime parameters (these constrain the final speaker count)
            runtime_params = {
                "min_speakers": getattr(self.config, 'diarization_min_speakers', None),
                "max_speakers": getattr(self.config, 'diarization_max_speakers', None)
            }
            # Filter out None values to avoid passing them to the pipeline
            runtime_params = {k: v for k, v in runtime_params.items() if v is not None}

            logger.info(f"🎤 Running diarization with runtime constraints: {runtime_params}")

            # 3. Run diarization on the audio file
            diarization = self.pyannote_pipeline(audio_path, **runtime_params)
            
            if not diarization:
                logger.error("❌ Diarization returned no result.")
                return None
            
            # 4. Save raw diarization output for debugging
            self._save_raw_diarization_results(diarization, audio_path)
            
            # 5. Assign speakers to the text segments based on time overlap
            result_segments = self._assign_speakers_by_overlap(segments, diarization)

            # 6. Clean up memory
            del diarization
            gc.collect()

            logger.info("✅ Speaker diarization completed successfully.")
            return {"segments": result_segments}

        except Exception as e:
            logger.error(f"❌ An error occurred during speaker diarization: {e}", exc_info=True)
            return None

    def _safe_time_to_seconds(self, time_value: Any) -> float:
        """
        Safely converts various time formats (timedelta, int, float, str) to float seconds.
        """
        try:
            if hasattr(time_value, 'total_seconds'):
                return time_value.total_seconds()
            return float(time_value)
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Failed to convert time value '{time_value}' to seconds: {e}")
            return 0.0

    def _assign_speakers_by_overlap(self, segments: List[Dict], diarization) -> List[Dict]:
        """
        Assigns speakers to each segment based on overlapping diarization turns.
        This handles cases where one dialogue line might involve multiple speakers.

        Args:
            segments: A list of transcribed segments from an SRT file.
            diarization: A Pyannote Annotation object with speaker turns.

        Returns:
            The list of segments, each with 'speakers' and 'all_speakers' keys.
        """
        diarization_turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_turns.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.end - turn.start
            })
        
        logger.info(f"🎯 Assigning speakers to {len(segments)} segments using overlap analysis...")
        
        for segment in segments:
            seg_start = self._safe_time_to_seconds(segment["start"])
            seg_end = self._safe_time_to_seconds(segment["end"])
            seg_duration = seg_end - seg_start
            
            overlapping_speakers = []
            for turn in diarization_turns:
                overlap_start = max(seg_start, turn['start'])
                overlap_end = min(seg_end, turn['end'])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    overlap_percentage = (overlap_duration / seg_duration) * 100 if seg_duration > 0 else 0
                    overlapping_speakers.append({
                        'speaker': turn['speaker'],
                        'overlap_duration': overlap_duration,
                        'overlap_percentage': overlap_percentage,
                    })
            
            overlapping_speakers.sort(key=lambda x: x['overlap_duration'], reverse=True)
            
            segment['speakers'] = overlapping_speakers
            segment['all_speakers'] = list(set(s['speaker'] for s in overlapping_speakers))

        return segments
    
    def _save_raw_diarization_results(self, diarization, audio_path: str) -> None:
        """
        Saves raw diarization results to text and JSON files for debugging.
        
        Args:
            diarization: Pyannote diarization annotation object.
            audio_path: Path to the audio file for naming the output files.
        """
        try:
            audio_dir = os.path.dirname(audio_path)
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Save as human-readable text file
            txt_output_path = os.path.join(audio_dir, f"{audio_name}_raw_diarization.txt")
            logger.info(f"💾 Saving raw diarization results to: {txt_output_path}")
            with open(txt_output_path, 'w', encoding='utf-8') as f:
                diarization.write_rttm(f)

            # Save as structured JSON file
            json_output_path = os.path.join(audio_dir, f"{audio_name}_raw_diarization.json")
            diarization_list = [
                {'start': turn.start, 'end': turn.end, 'speaker': speaker, 'duration': turn.duration}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(diarization_list, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Raw diarization JSON saved to: {json_output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save raw diarization results: {e}")

    def _annotation_to_df(self, annotation) -> pd.DataFrame:
        """Converts a pyannote annotation object to a pandas DataFrame."""
        rows = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            rows.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        return pd.DataFrame(rows)