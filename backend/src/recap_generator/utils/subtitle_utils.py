"""
Subtitle processing utilities for recap generation.

This module provides functions for parsing SRT files, extracting subtitle sequences,
and processing subtitles for recap clips. Extends existing subtitle functionality.
"""

import os
import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ...utils.logger_utils import setup_logging
from ..exceptions.recap_exceptions import SubtitleProcessingError
from ..models.event_models import SubtitleSequence
from ...narrative_storage_management.narrative_models import DialogueLine

logger = setup_logging(__name__)


@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    speaker: Optional[str] = None


class SubtitleUtils:
    """Utility class for subtitle processing operations."""
    
    @staticmethod
    def parse_srt_file(srt_path: str) -> List[SubtitleEntry]:
        """
        Parse an SRT file and return list of subtitle entries.
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            List of SubtitleEntry objects
        """
        try:
            if not os.path.exists(srt_path):
                raise SubtitleProcessingError(
                    f"SRT file not found: {srt_path}",
                    subtitle_file=srt_path
                )
            
            entries = []
            
            with open(srt_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split by double newlines to get individual subtitle blocks
            blocks = re.split(r'\n\s*\n', content.strip())
            
            for block in blocks:
                if not block.strip():
                    continue
                
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue  # Skip malformed entries
                
                try:
                    # Parse index
                    index = int(lines[0])
                    
                    # Parse timestamp line
                    timestamp_line = lines[1]
                    start_str, end_str = timestamp_line.split(' --> ')
                    
                    start_time = SubtitleUtils._parse_srt_timestamp(start_str.strip())
                    end_time = SubtitleUtils._parse_srt_timestamp(end_str.strip())
                    
                    # Parse text (may span multiple lines)
                    text_lines = lines[2:]
                    text = ' '.join(text_lines).strip()
                    
                    # Check for speaker identification in text
                    speaker = SubtitleUtils._extract_speaker_from_text(text)
                    
                    entries.append(SubtitleEntry(
                        index=index,
                        start_time=start_time,
                        end_time=end_time,
                        text=text,
                        speaker=speaker
                    ))
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"⚠️ Skipping malformed subtitle entry: {block[:50]}... Error: {e}")
                    continue
            
            logger.info(f"✅ Parsed {len(entries)} subtitle entries from {srt_path}")
            return entries
            
        except Exception as e:
            raise SubtitleProcessingError(
                f"Failed to parse SRT file: {e}",
                subtitle_file=srt_path
            )
    
    @staticmethod
    def _parse_srt_timestamp(timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
        try:
            # Handle both comma and dot as decimal separator
            timestamp = timestamp.replace(',', '.')
            
            # Split into time and milliseconds
            if '.' in timestamp:
                time_part, ms_part = timestamp.rsplit('.', 1)
                milliseconds = float('0.' + ms_part)
            else:
                time_part = timestamp
                milliseconds = 0
            
            # Parse time part
            time_components = time_part.split(':')
            if len(time_components) != 3:
                raise ValueError(f"Invalid timestamp format: {timestamp}")
            
            hours, minutes, seconds = map(int, time_components)
            
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
            return total_seconds
            
        except (ValueError, IndexError) as e:
            raise SubtitleProcessingError(f"Invalid timestamp format: {timestamp}")
    
    @staticmethod
    def _extract_speaker_from_text(text: str) -> Optional[str]:
        """Extract speaker name from subtitle text if present."""
        # Look for patterns like "SPEAKER:" or "(Speaker)" or "[Speaker]"
        patterns = [
            r'^([A-Z][A-Z\s]+):\s*',  # SPEAKER: text
            r'^\(([^)]+)\):\s*',      # (Speaker): text  
            r'^\[([^\]]+)\]:\s*',     # [Speaker]: text
            r'^([A-Z][a-z]+):\s*'     # Speaker: text
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip())
            if match:
                speaker = match.group(1).strip()
                # Basic validation - speaker names shouldn't be too long or contain weird characters
                if len(speaker) <= 30 and re.match(r'^[A-Za-z\s\-\.]+$', speaker):
                    return speaker
        
        return None
    
    @staticmethod
    def extract_subtitles_for_timerange(entries: List[SubtitleEntry],
                                      start_time: float,
                                      end_time: float,
                                      padding_seconds: float = 0.5) -> List[SubtitleEntry]:
        """
        Extract subtitle entries that fall within a time range.
        
        Args:
            entries: List of subtitle entries
            start_time: Start time in seconds
            end_time: End time in seconds  
            padding_seconds: Extra padding to include nearby subtitles
            
        Returns:
            List of relevant subtitle entries
        """
        try:
            # Expand time range with padding
            padded_start = max(0, start_time - padding_seconds)
            padded_end = end_time + padding_seconds
            
            relevant_entries = []
            
            for entry in entries:
                # Check if subtitle overlaps with time range
                if (entry.start_time <= padded_end and entry.end_time >= padded_start):
                    relevant_entries.append(entry)
            
            # Sort by start time
            relevant_entries.sort(key=lambda x: x.start_time)
            
            logger.debug(f"🎯 Found {len(relevant_entries)} subtitles for range {start_time:.1f}-{end_time:.1f}s")
            return relevant_entries
            
        except Exception as e:
            raise SubtitleProcessingError(
                f"Failed to extract subtitles for timerange {start_time}-{end_time}: {e}",
                timestamp_range=(start_time, end_time)
            )
    
    @staticmethod
    def select_key_subtitle_sequence(entries: List[SubtitleEntry],
                                   target_lines: int = 5,
                                   max_lines: int = 7,
                                   min_lines: int = 2) -> SubtitleSequence:
        """
        Select a key sequence of subtitles that forms a coherent dialogue snippet.
        
        Args:
            entries: Available subtitle entries
            target_lines: Target number of subtitle lines
            max_lines: Maximum number of lines
            min_lines: Minimum number of lines
            
        Returns:
            SubtitleSequence object with selected lines
        """
        try:
            if len(entries) == 0:
                raise SubtitleProcessingError("No subtitle entries provided for sequence selection")
            
            if len(entries) <= target_lines:
                # Use all available entries
                selected_entries = entries
            else:
                # Try to find the best sequence
                selected_entries = SubtitleUtils._find_best_subtitle_sequence(
                    entries, target_lines, max_lines, min_lines
                )
            
            if len(selected_entries) < min_lines:
                logger.warning(f"⚠️ Selected sequence has only {len(selected_entries)} lines (min: {min_lines})")
            
            # Extract information from selected entries
            lines = [entry.text for entry in selected_entries]
            start_time = selected_entries[0].start_time
            end_time = selected_entries[-1].end_time
            speakers = list(set(entry.speaker for entry in selected_entries if entry.speaker))
            
            # Calculate quality metrics
            dialogue_quality = SubtitleUtils._assess_dialogue_quality(selected_entries)
            narrative_completeness = SubtitleUtils._assess_narrative_completeness(selected_entries)
            
            return SubtitleSequence(
                event_id="",  # Will be set by caller
                lines=lines,
                start_time=start_time,
                end_time=end_time,
                speakers=speakers,
                dialogue_quality=dialogue_quality,
                narrative_completeness=narrative_completeness
            )
            
        except Exception as e:
            raise SubtitleProcessingError(f"Failed to select subtitle sequence: {e}")
    
    @staticmethod
    def _find_best_subtitle_sequence(entries: List[SubtitleEntry],
                                   target_lines: int,
                                   max_lines: int,
                                   min_lines: int) -> List[SubtitleEntry]:
        """Find the best contiguous sequence of subtitles."""
        best_sequence = []
        best_score = 0
        
        # Try different starting positions and lengths
        for start_idx in range(len(entries)):
            for length in range(min_lines, min(max_lines + 1, len(entries) - start_idx + 1)):
                sequence = entries[start_idx:start_idx + length]
                score = SubtitleUtils._score_subtitle_sequence(sequence, target_lines)
                
                if score > best_score:
                    best_score = score
                    best_sequence = sequence
        
        return best_sequence if best_sequence else entries[:min(target_lines, len(entries))]
    
    @staticmethod
    def _score_subtitle_sequence(entries: List[SubtitleEntry], target_lines: int) -> float:
        """Score a subtitle sequence based on various quality factors."""
        if not entries:
            return 0
        
        score = 0
        
        # Length score (prefer target length)
        length_diff = abs(len(entries) - target_lines)
        length_score = max(0, 1 - (length_diff / target_lines))
        score += length_score * 0.3
        
        # Continuity score (prefer contiguous time periods)
        total_duration = entries[-1].end_time - entries[0].start_time
        subtitle_duration = sum(entry.end_time - entry.start_time for entry in entries)
        continuity_score = subtitle_duration / total_duration if total_duration > 0 else 0
        score += continuity_score * 0.2
        
        # Content quality score (prefer non-empty, meaningful text)
        text_lengths = [len(entry.text.strip()) for entry in entries]
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        content_score = min(1.0, avg_text_length / 50)  # Normalize to 50 characters
        score += content_score * 0.3
        
        # Speaker diversity score (prefer dialogue with multiple speakers)
        speakers = set(entry.speaker for entry in entries if entry.speaker)
        speaker_diversity = min(1.0, len(speakers) / 2)  # Normalize to 2 speakers
        score += speaker_diversity * 0.2
        
        return score
    
    @staticmethod
    def _assess_dialogue_quality(entries: List[SubtitleEntry]) -> float:
        """Assess the quality of dialogue in subtitle entries."""
        if not entries:
            return 0
        
        quality_factors = []
        
        # Check for meaningful text (not just sound effects)
        meaningful_text_ratio = sum(
            1 for entry in entries 
            if entry.text and not re.match(r'^\[.*\]$|^\(.*\)$', entry.text.strip())
        ) / len(entries)
        quality_factors.append(meaningful_text_ratio)
        
        # Check text length distribution
        text_lengths = [len(entry.text.strip()) for entry in entries if entry.text.strip()]
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            length_score = min(1.0, avg_length / 40)  # Normalize to 40 characters
            quality_factors.append(length_score)
        
        # Check for speaker identification
        speaker_ratio = sum(1 for entry in entries if entry.speaker) / len(entries)
        quality_factors.append(speaker_ratio)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0
    
    @staticmethod
    def _assess_narrative_completeness(entries: List[SubtitleEntry]) -> float:
        """Assess how complete the narrative snippet is."""
        if not entries:
            return 0
        
        completeness_factors = []
        
        # Check for sentence completeness
        combined_text = ' '.join(entry.text for entry in entries if entry.text)
        sentence_endings = combined_text.count('.') + combined_text.count('!') + combined_text.count('?')
        sentence_ratio = min(1.0, sentence_endings / max(1, len(entries) // 2))
        completeness_factors.append(sentence_ratio)
        
        # Check for dialogue flow (questions and responses)
        questions = sum(1 for entry in entries if '?' in entry.text)
        if questions > 0:
            # Look for responses after questions
            has_responses = any(
                i < len(entries) - 1 and entries[i].speaker != entries[i + 1].speaker
                for i, entry in enumerate(entries) if '?' in entry.text
            )
            response_score = 1.0 if has_responses else 0.5
            completeness_factors.append(response_score)
        
        # Check temporal continuity (no large gaps)
        if len(entries) > 1:
            gaps = [
                entries[i + 1].start_time - entries[i].end_time
                for i in range(len(entries) - 1)
            ]
            max_gap = max(gaps) if gaps else 0
            gap_score = max(0, 1 - (max_gap / 10))  # Penalize gaps > 10 seconds
            completeness_factors.append(gap_score)
        
        return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0


def convert_dialogue_lines_to_subtitle_entries(dialogue_lines: List[DialogueLine]) -> List[SubtitleEntry]:
    """
    Convert DialogueLine objects to SubtitleEntry objects for processing.
    
    Args:
        dialogue_lines: List of DialogueLine objects from existing system
        
    Returns:
        List of SubtitleEntry objects
    """
    try:
        entries = []
        
        for line in dialogue_lines:
            entry = SubtitleEntry(
                index=line.index,
                start_time=line.start_time,
                end_time=line.end_time,
                text=line.text,
                speaker=line.speaker
            )
            entries.append(entry)
        
        logger.debug(f"✅ Converted {len(dialogue_lines)} DialogueLines to SubtitleEntries")
        return entries
        
    except Exception as e:
        raise SubtitleProcessingError(f"Failed to convert DialogueLines: {e}")


def validate_subtitle_file(srt_path: str) -> Dict[str, Any]:
    """
    Validate an SRT file and return quality metrics.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        Dictionary with validation results
    """
    try:
        if not os.path.exists(srt_path):
            return {
                'valid': False,
                'error': 'File does not exist',
                'entry_count': 0,
                'duration': 0
            }
        
        entries = SubtitleUtils.parse_srt_file(srt_path)
        
        if not entries:
            return {
                'valid': False,
                'error': 'No subtitle entries found',
                'entry_count': 0,
                'duration': 0
            }
        
        # Calculate metrics
        total_duration = entries[-1].end_time - entries[0].start_time
        has_speakers = sum(1 for entry in entries if entry.speaker) / len(entries)
        avg_text_length = sum(len(entry.text) for entry in entries) / len(entries)
        
        # Check for common issues
        issues = []
        
        # Check for overlapping subtitles
        overlaps = 0
        for i in range(len(entries) - 1):
            if entries[i].end_time > entries[i + 1].start_time:
                overlaps += 1
        
        if overlaps > len(entries) * 0.1:  # More than 10% overlaps
            issues.append(f"High number of overlapping subtitles: {overlaps}")
        
        # Check for very short or long subtitles
        short_subtitles = sum(1 for entry in entries if entry.end_time - entry.start_time < 0.5)
        long_subtitles = sum(1 for entry in entries if entry.end_time - entry.start_time > 10)
        
        if short_subtitles > len(entries) * 0.2:
            issues.append(f"Many very short subtitles: {short_subtitles}")
        
        if long_subtitles > len(entries) * 0.1:
            issues.append(f"Many very long subtitles: {long_subtitles}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'entry_count': len(entries),
            'duration': total_duration,
            'speaker_ratio': has_speakers,
            'avg_text_length': avg_text_length,
            'overlap_count': overlaps
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'entry_count': 0,
            'duration': 0
        }
    
    @staticmethod
    def timestamp_to_seconds(timestamp: str) -> float:
        """Convert SRT timestamp to seconds."""
        try:
            # Handle both HH:MM:SS,mmm and HH:MM:SS formats
            if ',' in timestamp:
                time_part, ms_part = timestamp.split(',')
                milliseconds = float(ms_part) / 1000
            else:
                time_part = timestamp
                milliseconds = 0.0
            
            # Parse time part
            parts = time_part.split(':')
            hours, minutes, seconds = map(int, parts)
            
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
            return total_seconds
        except:
            return 0.0
    
    @staticmethod
    def seconds_to_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds % 1) * 1000)
            
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
        except:
            return "00:00:00,000"
