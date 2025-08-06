"""
Subtitle Speaker Identification Module for SEMAMORPH.

This module provides functionality for:
1. Parsing SRT subtitle files
2. Identifying speakers using LLM analysis  
3. Extracting faces from video at dialogue midpoints
4. Generating face embeddings
5. Clustering faces and associating with speakers
6. Resolving uncertain speaker assignments

Main entry point is the SpeakerIdentificationPipeline class.
"""

from .srt_parser import SRTParser
from .speaker_identifier import SpeakerIdentifier
from .speaker_character_validator import SpeakerCharacterValidator
from .subtitle_face_extractor import SubtitleFaceExtractor
from .subtitle_face_embedder import SubtitleFaceEmbedder
from .face_clustering_system import FaceClusteringSystem
from .speaker_identification_pipeline import SpeakerIdentificationPipeline, run_face_clustering_only
from .pipeline_factory import run_speaker_identification_pipeline_compat as run_speaker_identification_pipeline

__all__ = [
    'SRTParser',
    'SpeakerIdentifier',
    'SpeakerCharacterValidator', 
    'SubtitleFaceExtractor',
    'SubtitleFaceEmbedder',
    'FaceClusteringSystem',
    'SpeakerIdentificationPipeline',
    'run_speaker_identification_pipeline',
    'run_face_clustering_only'
]
