"""
Audio processing module for SEMAMORPH.
Handles vocal extraction, audio preprocessing, and audio analysis.
"""

from .demucs_vocal_extractor import DemucsVocalExtractor, extract_vocals_for_diarization

__all__ = [
    'DemucsVocalExtractor',
    'extract_vocals_for_diarization'
] 