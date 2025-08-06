"""
Subtitle Processor Service for recap generation.

This service processes subtitle files to extract optimal dialogue sequences
for selected recap events, ensuring narrative completeness and quality.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..ai_models.ai_models import get_llm, LLMType
from ..utils.logger_utils import setup_logging
from .exceptions.recap_exceptions import LLMServiceError, SubtitleProcessingError
from .models.event_models import VectorEvent, SubtitleSequence, SubtitleEntry
from .utils.subtitle_utils import SubtitleUtils, SubtitleEntry as UtilsSubtitleEntry
from ..path_handler import PathHandler

logger = setup_logging(__name__)


class SubtitleProcessorService:
    """Service for processing subtitles to create optimal dialogue sequences for recap events."""
    
    def __init__(self, path_handler: PathHandler):
        self.path_handler = path_handler
        self.llm = get_llm(LLMType.INTELLIGENT)
        self.subtitle_utils = SubtitleUtils()
        
    def process_event_subtitles(self, selected_events: List[VectorEvent]) -> List[SubtitleSequence]:
        """
        Process subtitle files to extract optimal dialogue sequences for selected events.
        
        Args:
            selected_events: List of events selected for recap inclusion
            
        Returns:
            List of SubtitleSequence objects with optimized dialogue
        """
        try:
            logger.info(f"📝 Processing subtitles for {len(selected_events)} events")
            
            # Process each event with its own subtitle file
            subtitle_sequences = []
            
            for i, event in enumerate(selected_events):
                logger.debug(f"🎬 Processing event {i+1}/{len(selected_events)}: {event.arc_title}")
                
                try:
                    # Load subtitle file from the event's source episode
                    subtitle_entries = self._load_subtitle_file_for_event(event)
                    
                    # Extract relevant subtitle entries for this event
                    event_subtitles = self._extract_event_subtitles(event, subtitle_entries)
                    
                    if not event_subtitles:
                        logger.warning(f"⚠️ No subtitles found for event {event.id}")
                        continue
                    
                    # Create initial subtitle sequence
                    sequence = self._create_subtitle_sequence(event, event_subtitles)
                    
                    # Enhance sequence using LLM
                    enhanced_sequence = self._enhance_subtitle_sequence(sequence)
                    
                    subtitle_sequences.append(enhanced_sequence)
                    logger.debug(f"✅ Event {i+1}: {len(enhanced_sequence.lines)} subtitle lines, {enhanced_sequence.duration:.1f}s")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Event {i+1} subtitle processing failed: {e}")
                    continue
            
            logger.info(f"✅ Subtitle processing completed: {len(subtitle_sequences)}/{len(selected_events)} events processed")
            
            return subtitle_sequences
            
        except Exception as e:
            raise SubtitleProcessingError(
                f"Subtitle processing failed: {e}",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def optimize_sequence_timing(self, sequences: List[SubtitleSequence]) -> List[SubtitleSequence]:
        """
        Optimize subtitle sequence timing for natural dialogue boundaries.
        
        Args:
            sequences: List of subtitle sequences to optimize
            
        Returns:
            List of optimized subtitle sequences
        """
        try:
            logger.info(f"⚙️ Optimizing timing for {len(sequences)} subtitle sequences")
            
            optimized_sequences = []
            
            for i, sequence in enumerate(sequences):
                try:
                    # Use LLM to analyze and optimize the sequence
                    optimized = self._optimize_single_sequence(sequence)
                    optimized_sequences.append(optimized)
                    
                    timing_change = optimized.duration - sequence.duration
                    logger.debug(f"🎯 Sequence {i+1}: {timing_change:+.1f}s timing adjustment")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Sequence {i+1} optimization failed: {e}, using original")
                    optimized_sequences.append(sequence)
            
            total_duration = sum(seq.duration for seq in optimized_sequences)
            logger.info(f"✅ Timing optimization completed: {total_duration:.1f}s total duration")
            
            return optimized_sequences
            
        except Exception as e:
            raise LLMServiceError(
                f"Sequence timing optimization failed: {e}",
                service_name="SubtitleProcessorService.optimize_sequence_timing",
                series=self.path_handler.get_series(),
                season=self.path_handler.get_season(),
                episode=self.path_handler.get_episode()
            )
    
    def _load_subtitle_file(self) -> List[UtilsSubtitleEntry]:
        """Load and parse the enhanced subtitle file."""
        try:
            srt_path = self.path_handler.get_possible_speakers_srt_path()
            
            if not os.path.exists(srt_path):
                raise SubtitleProcessingError(
                    f"Subtitle file not found: {srt_path}",
                    subtitle_file=srt_path
                )
            
            # Parse subtitle file
            entries = self.subtitle_utils.parse_srt_file(srt_path)
            
            if not entries:
                raise SubtitleProcessingError(
                    "No subtitle entries found in file",
                    subtitle_file=srt_path
                )
            
            logger.info(f"📖 Loaded {len(entries)} subtitle entries from {os.path.basename(srt_path)}")
            logger.debug(f"📊 Duration: {entries[-1].end_time - entries[0].start_time:.1f}s")
            
            return entries
            
        except Exception as e:
            if isinstance(e, SubtitleProcessingError):
                raise
            else:
                raise SubtitleProcessingError(
                    f"Failed to load subtitle file: {e}",
                )

    def _load_subtitle_file_for_event(self, event: VectorEvent) -> List[UtilsSubtitleEntry]:
        """Load subtitle file from the event's source episode."""
        try:
            # Get event's episode information
            event_series = getattr(event, 'series', self.path_handler.get_series())
            event_season = getattr(event, 'season', self.path_handler.get_season())
            event_episode = getattr(event, 'episode', self.path_handler.get_episode())
            
            # Construct path to the event's source subtitle file
            srt_filename = f"{event_series}{event_season}{event_episode}_possible_speakers.srt"
            srt_path = os.path.join(
                self.path_handler.base_dir,
                event_series,
                event_season,
                event_episode,
                srt_filename
            )
            
            logger.debug(f"🎯 Loading subtitles from {event_series} {event_season} {event_episode}")
            logger.debug(f"📁 Subtitle path: {srt_path}")
            
            if not os.path.exists(srt_path):
                raise SubtitleProcessingError(
                    f"Event source subtitle file not found: {srt_path}",
                    subtitle_file=srt_path
                )
            
            # Parse subtitle file
            entries = self.subtitle_utils.parse_srt_file(srt_path)
            
            if not entries:
                raise SubtitleProcessingError(
                    "No subtitle entries found in event source file",
                    subtitle_file=srt_path
                )
            
            logger.debug(f"📖 Loaded {len(entries)} subtitle entries from event source {os.path.basename(srt_path)}")
            
            return entries
            
        except Exception as e:
            if isinstance(e, SubtitleProcessingError):
                raise
            else:
                raise SubtitleProcessingError(
                    f"Failed to load event source subtitle file: {e}",
                    subtitle_file=self.path_handler.get_possible_speakers_srt_path()
                )
    
    def _extract_event_subtitles(self, event: VectorEvent, 
                               subtitle_entries: List[UtilsSubtitleEntry]) -> List[UtilsSubtitleEntry]:
        """Extract subtitle entries that match the event's timestamp range."""
        
        # Parse event timestamps
        start_seconds = self._parse_timestamp_to_seconds(event.start_timestamp)
        end_seconds = self._parse_timestamp_to_seconds(event.end_timestamp)
        
        # Find overlapping subtitles
        matching_subtitles = []
        for subtitle in subtitle_entries:
            if subtitle.start_time <= end_seconds and subtitle.end_time >= start_seconds:
                matching_subtitles.append(subtitle)
        
        logger.debug(f"🎯 Found {len(matching_subtitles)} subtitles for range {start_seconds:.1f}s-{end_seconds:.1f}s")
        
        return matching_subtitles
    
    def _create_subtitle_sequence(self, event: VectorEvent, 
                                subtitle_entries: List[UtilsSubtitleEntry]) -> SubtitleSequence:
        """Create a SubtitleSequence from event and subtitle entries."""
        
        if not subtitle_entries:
            raise SubtitleProcessingError(f"No subtitle entries provided for event {event.id}")
        
        # Calculate sequence timing
        start_time = min(entry.start_time for entry in subtitle_entries)
        end_time = max(entry.end_time for entry in subtitle_entries)
        duration = end_time - start_time
        
        # Extract text lines
        lines = [entry.text for entry in subtitle_entries]
        
        # Extract speakers (basic extraction from text)
        speakers = []
        for line in lines:
            if ':' in line and '/' not in line:  # Simple speaker detection
                speaker = line.split(':')[0].strip()
                if speaker not in speakers:
                    speakers.append(speaker)
        
        # Convert utils SubtitleEntry to model SubtitleEntry for storage
        original_entries = [
            SubtitleEntry(
                index=i,
                start_time=entry.start_time,
                end_time=entry.end_time,
                text=entry.text
            ) for i, entry in enumerate(subtitle_entries)
        ]
        
        # Create sequence
        sequence = SubtitleSequence(
            event_id=event.id,
            lines=lines,
            start_time=start_time,
            end_time=end_time,
            speakers=speakers,
            dialogue_quality=0.8,  # Default quality
            narrative_completeness=0.8,  # Default completeness
            has_dialogue=len(lines) > 0,
            original_entries=original_entries
        )
        
        return sequence
    
    def _enhance_subtitle_sequence(self, sequence: SubtitleSequence) -> SubtitleSequence:
        """Enhance subtitle sequence using LLM to select best dialogue for recap."""
        
        try:
            # If sequence is already short enough, return as-is
            if len(sequence.lines) <= 3 and sequence.duration <= 8.0:
                logger.debug(f"✅ Sequence already optimal: {len(sequence.lines)} lines, {sequence.duration:.1f}s")
                return sequence
            
            # Use LLM to select the most impactful subtitle lines
            selected_lines = self._llm_select_best_subtitles(sequence)
            
            if not selected_lines:
                logger.warning(f"⚠️ LLM selection failed, using original sequence")
                return sequence
            
            # Create optimized sequence with selected lines only
            optimized_sequence = self._create_optimized_sequence(sequence, selected_lines)
            
            logger.debug(f"✅ Enhanced sequence for event {sequence.event_id}: {len(sequence.lines)} → {len(optimized_sequence.lines)} lines, {sequence.duration:.1f}s → {optimized_sequence.duration:.1f}s")
            return optimized_sequence
            
        except Exception as e:
            logger.warning(f"⚠️ Sequence enhancement failed: {e}")
            return sequence
    
    def _llm_select_best_subtitles(self, sequence: SubtitleSequence) -> List[int]:
        """Use LLM to select the best subtitle lines for a recap clip."""
        
        try:
            # Create prompt for LLM selection
            prompt = f"""You are creating a "Previously On..." recap for a TV show. 

You have these subtitle lines from a key scene:
{chr(10).join([f"{i+1}. {line}" for i, line in enumerate(sequence.lines)])}

Select the 2-3 most impactful and essential lines that capture the key moment for a recap. 
The selected lines should:
- Be consecutive or close together for smooth video editing
- Contain the most important dialogue or revelation
- Work well in a brief "Previously On" context
- Total duration should be 3-8 seconds maximum

Respond with ONLY the line numbers (e.g., "2,3,4" or "1,2"):"""

            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse line numbers from response
            line_indices = []
            for part in response_text.strip().split(','):
                try:
                    index = int(part.strip()) - 1  # Convert to 0-based index
                    if 0 <= index < len(sequence.lines):
                        line_indices.append(index)
                except ValueError:
                    continue
            
            # Validate selection
            if not line_indices:
                logger.warning("⚠️ LLM returned no valid line selections")
                return []
            
            if len(line_indices) > 4:  # Limit to max 4 lines
                line_indices = line_indices[:4]
            
            logger.debug(f"🎯 LLM selected lines: {[i+1 for i in line_indices]} out of {len(sequence.lines)}")
            return line_indices
            
        except Exception as e:
            logger.warning(f"⚠️ LLM subtitle selection failed: {e}")
            return []
    
    def _create_optimized_sequence(self, original_sequence: SubtitleSequence, 
                                 selected_indices: List[int]) -> SubtitleSequence:
        """Create an optimized sequence using only the selected subtitle lines with precise timing."""
        
        if not selected_indices:
            return original_sequence
        
        # Get selected lines
        selected_lines = [original_sequence.lines[i] for i in selected_indices]
        
        # Calculate precise timing using original subtitle entries
        if original_sequence.original_entries and len(original_sequence.original_entries) > max(selected_indices):
            # Use precise timing from original subtitle entries
            selected_entries = [original_sequence.original_entries[i] for i in selected_indices]
            
            # Calculate precise start and end times
            precise_start = min(entry.start_time for entry in selected_entries)
            precise_end = max(entry.end_time for entry in selected_entries)
            
            # Add small padding for smoother cuts (±0.3 seconds)
            padding = 0.3
            optimized_start = max(0, precise_start - padding)
            optimized_end = precise_end + padding
            
            logger.debug(f"🎯 Precise timing: {precise_start:.1f}s-{precise_end:.1f}s → {optimized_start:.1f}s-{optimized_end:.1f}s (with padding)")
            
        else:
            # Fallback to estimation if no original entries available
            logger.warning("⚠️ No original entries available, using estimated timing")
            lines_per_second = len(original_sequence.lines) / original_sequence.duration if original_sequence.duration > 0 else 1
            
            first_line_index = min(selected_indices)
            estimated_start_offset = first_line_index / lines_per_second if lines_per_second > 0 else 0
            estimated_duration = len(selected_lines) / lines_per_second if lines_per_second > 0 else 3.0
            estimated_duration = max(2.0, min(8.0, estimated_duration))
            
            optimized_start = original_sequence.start_time + estimated_start_offset
            optimized_end = optimized_start + estimated_duration
        
        # Create optimized sequence
        optimized_sequence = SubtitleSequence(
            event_id=original_sequence.event_id,
            lines=selected_lines,
            start_time=optimized_start,
            end_time=optimized_end,
            speakers=original_sequence.speakers,
            dialogue_quality=0.9,  # Higher quality due to LLM selection
            narrative_completeness=0.9,
            has_dialogue=True,
            selection_reasoning=f"LLM selected lines {[i+1 for i in selected_indices]} from {len(original_sequence.lines)} total"
        )
        
        return optimized_sequence
    
    def _process_single_event(self, event: VectorEvent, 
                             subtitle_entries: List[SubtitleEntry]) -> Optional[SubtitleSequence]:
        """Process subtitles for a single event."""
        
        # Check if event has valid timestamps
        if not event.has_valid_timestamps:
            logger.warning(f"⚠️ Event '{event.id}' lacks valid timestamps, skipping")
            return None
        
        try:
            # Parse timestamps
            start_time = self._parse_timestamp_to_seconds(event.start_timestamp)
            end_time = self._parse_timestamp_to_seconds(event.end_timestamp)
            
            if start_time >= end_time:
                logger.warning(f"⚠️ Event '{event.id}' has invalid time range: {start_time}-{end_time}")
                return None
            
            # Extract relevant subtitle entries
            relevant_entries = self.subtitle_utils.extract_subtitles_for_timerange(
                subtitle_entries,
                start_time,
                end_time,
                padding_seconds=1.0  # Add some padding for context
            )
            
            if not relevant_entries:
                logger.warning(f"⚠️ No subtitles found for event '{event.id}' timerange {start_time}-{end_time}")
                return None
            
            # Select optimal subtitle sequence
            sequence = self.subtitle_utils.select_key_subtitle_sequence(
                relevant_entries,
                target_lines=5,
                max_lines=7,
                min_lines=2
            )
            
            # Set event ID
            sequence.event_id = event.id
            
            # Use LLM to enhance sequence selection if needed
            enhanced_sequence = self._enhance_sequence_with_llm(sequence, event)
            
            return enhanced_sequence
            
        except Exception as e:
            logger.error(f"❌ Failed to process event '{event.id}': {e}")
            return None
    
    def _parse_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp format to seconds."""
        try:
            # Handle format: HH:MM:SS,mmm
            time_part, ms_part = timestamp.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            
            return h * 3600 + m * 60 + s + ms / 1000.0
            
        except (ValueError, IndexError) as e:
            raise SubtitleProcessingError(f"Invalid timestamp format: {timestamp}")
    
    def _enhance_sequence_with_llm(self, sequence: SubtitleSequence, 
                                  event: VectorEvent) -> SubtitleSequence:
        """Use LLM to enhance subtitle sequence selection and timing."""
        try:
            # Create analysis prompt
            prompt = self._create_sequence_enhancement_prompt(sequence, event)
            
            # Get LLM analysis
            response = self.llm.invoke(prompt)
            
            # Parse LLM response
            enhancements = self._parse_enhancement_response(response.content)
            
            # Apply enhancements
            enhanced_sequence = self._apply_enhancements(sequence, enhancements)
            
            return enhanced_sequence
            
        except Exception as e:
            logger.warning(f"⚠️ LLM enhancement failed for event '{event.id}': {e}, using original sequence")
            return sequence
    
    def _create_sequence_enhancement_prompt(self, sequence: SubtitleSequence, 
                                          event: VectorEvent) -> str:
        """Create prompt for LLM sequence enhancement."""
        
        lines_text = "\\n".join([f"{i+1}. {line}" for i, line in enumerate(sequence.lines)])
        
        return f"""You are analyzing subtitle sequences for a TV episode recap. Your task is to evaluate and potentially improve a selected dialogue sequence.

**Event Context:**
- Arc: {event.arc_title}
- Type: {event.arc_type}
- Content: {event.content}
- Characters: {', '.join(event.main_characters)}

**Selected Subtitle Sequence:**
Duration: {sequence.duration:.1f} seconds
Lines:
{lines_text}

**Your Task:**
Analyze this sequence and provide recommendations for:

1. **Line Selection**: Should any lines be removed or are important lines missing?
2. **Narrative Completeness**: Does this sequence tell a complete mini-story?
3. **Dialogue Quality**: Is the dialogue clear and meaningful?
4. **Character Context**: Are the important characters well represented?
5. **Emotional Impact**: Does the sequence convey the right emotional tone?

**Output Format (JSON):**
```json
{{
    "assessment": {{
        "narrative_completeness": 0.8,
        "dialogue_quality": 0.9,
        "character_representation": 0.7,
        "emotional_impact": 0.8,
        "overall_quality": 0.8
    }},
    "recommendations": {{
        "remove_lines": [2, 4],
        "timing_adjustments": {{
            "start_offset": -0.5,
            "end_offset": 1.0
        }},
        "reasoning": "Brief explanation of recommendations"
    }},
    "alternative_focus": "If this sequence isn't optimal, what should we focus on instead?"
}}
```

Focus on creating the most effective 3-7 line sequence that provides clear context for the current episode."""
    
    def _parse_enhancement_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM enhancement response."""
        try:
            import re
            import json
            
            # Extract JSON from response
            json_match = re.search(r'```json\\s*({.*?})\\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'{.*}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")
            
            enhancements = json.loads(json_str)
            
            # Validate structure
            if 'assessment' not in enhancements:
                enhancements['assessment'] = {}
            if 'recommendations' not in enhancements:
                enhancements['recommendations'] = {}
            
            return enhancements
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️ Failed to parse enhancement response: {e}")
            return {
                'assessment': {'overall_quality': 0.7},
                'recommendations': {}
            }
    
    def _apply_enhancements(self, sequence: SubtitleSequence, 
                          enhancements: Dict[str, Any]) -> SubtitleSequence:
        """Apply LLM-suggested enhancements to subtitle sequence."""
        
        enhanced_sequence = SubtitleSequence(
            event_id=sequence.event_id,
            lines=sequence.lines.copy(),
            start_time=sequence.start_time,
            end_time=sequence.end_time,
            speakers=sequence.speakers.copy(),
            dialogue_quality=sequence.dialogue_quality,
            narrative_completeness=sequence.narrative_completeness
        )
        
        recommendations = enhancements.get('recommendations', {})
        assessment = enhancements.get('assessment', {})
        
        # Apply quality scores from LLM assessment
        if 'dialogue_quality' in assessment:
            enhanced_sequence.dialogue_quality = assessment['dialogue_quality']
        if 'narrative_completeness' in assessment:
            enhanced_sequence.narrative_completeness = assessment['narrative_completeness']
        
        # Apply line removals
        remove_lines = recommendations.get('remove_lines', [])
        if remove_lines:
            # Remove lines (1-indexed to 0-indexed)
            new_lines = [
                line for i, line in enumerate(enhanced_sequence.lines)
                if (i + 1) not in remove_lines
            ]
            if new_lines:  # Only apply if we still have lines left
                enhanced_sequence.lines = new_lines
                logger.debug(f"🔪 Removed {len(remove_lines)} lines based on LLM recommendation")
        
        # Apply timing adjustments
        timing_adjustments = recommendations.get('timing_adjustments', {})
        if timing_adjustments:
            start_offset = timing_adjustments.get('start_offset', 0)
            end_offset = timing_adjustments.get('end_offset', 0)
            
            new_start = max(0, enhanced_sequence.start_time + start_offset)
            new_end = enhanced_sequence.end_time + end_offset
            
            if new_end > new_start:
                enhanced_sequence.start_time = new_start
                enhanced_sequence.end_time = new_end
                logger.debug(f"⏰ Applied timing adjustments: {start_offset:+.1f}s start, {end_offset:+.1f}s end")
        
        # Add reasoning to sequence
        reasoning = recommendations.get('reasoning', '')
        if reasoning:
            enhanced_sequence.selection_reasoning = reasoning
        
        return enhanced_sequence
    
    def _optimize_single_sequence(self, sequence: SubtitleSequence) -> SubtitleSequence:
        """Optimize timing for a single subtitle sequence."""
        
        # For now, apply simple optimization rules
        # In a more sophisticated implementation, this would use more complex analysis
        
        optimized = SubtitleSequence(
            event_id=sequence.event_id,
            lines=sequence.lines.copy(),
            start_time=sequence.start_time,
            end_time=sequence.end_time,
            speakers=sequence.speakers.copy(),
            dialogue_quality=sequence.dialogue_quality,
            narrative_completeness=sequence.narrative_completeness,
            selection_reasoning=sequence.selection_reasoning
        )
        
        # Apply natural boundaries
        # Add small buffer at start for smooth transition
        optimized.start_time = max(0, optimized.start_time - 0.3)
        
        # Extend end time slightly for complete dialogue
        optimized.end_time = optimized.end_time + 0.5
        
        return optimized
