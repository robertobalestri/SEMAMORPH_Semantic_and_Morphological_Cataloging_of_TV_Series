"""
Subtitle Filtering Service using LLM intelligence.

This service uses LLMs to intelligently filter and select the most interesting,
contextually relevant, and dramatically effective subtitles from events
selected for recap inclusion.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ...ai_models.ai_models import get_llm, LLMType
from ...utils.logger_utils import setup_logging
from ..exceptions.recap_exceptions import LLMServiceError, SubtitleProcessingError
from ..models.recap_models import RecapEvent
from ..utils.subtitle_utils import SubtitleEntry

logger = setup_logging(__name__)


@dataclass
class FilteredSubtitle:
    """Represents a subtitle selected by LLM filtering."""
    subtitle: SubtitleEntry
    importance_score: float
    reasoning: str
    dramatic_value: str  # "high", "medium", "low"
    content_type: str   # "dialogue", "action", "emotional", "exposition"


class SubtitleFilteringService:
    """LLM-powered service for filtering and selecting optimal subtitles from events."""
    
    def __init__(self):
        self.llm = get_llm(LLMType.INTELLIGENT)
        
    def filter_event_subtitles(
        self, 
        event: RecapEvent, 
        subtitle_entries: List[SubtitleEntry],
        target_duration: float = 10.0,
        context: Optional[str] = None
    ) -> List[FilteredSubtitle]:
        """
        Use LLM to filter and select the most interesting subtitles from an event.
        
        Args:
            event: The recap event containing context information
            subtitle_entries: List of subtitle entries for the event timeframe
            target_duration: Target duration for the filtered subtitles
            context: Additional context about the episode or recap theme
            
        Returns:
            List of FilteredSubtitle objects ranked by importance
        """
        try:
            logger.info(f"🎭 Filtering subtitles for event: {event.content[:50]}...")
            
            if not subtitle_entries:
                logger.warning("⚠️ No subtitle entries provided for filtering")
                return []
            
            # Prepare subtitle text for LLM analysis
            subtitle_text = self._prepare_subtitle_text(subtitle_entries)
            
            # Create filtering prompt
            filtering_prompt = self._create_filtering_prompt(
                event, subtitle_text, target_duration, context
            )
            
            # Get LLM analysis
            llm_response = self._call_llm_for_filtering(filtering_prompt)
            
            # Parse LLM response and map back to subtitle entries
            filtered_subtitles = self._parse_filtering_response(
                llm_response, subtitle_entries
            )
            
            logger.info(f"✨ Selected {len(filtered_subtitles)} high-value subtitles from {len(subtitle_entries)} candidates")
            
            return filtered_subtitles
            
        except Exception as e:
            logger.error(f"❌ Subtitle filtering failed: {e}")
            raise SubtitleProcessingError(f"LLM subtitle filtering failed: {e}")
    
    def _prepare_subtitle_text(self, subtitle_entries: List[SubtitleEntry]) -> str:
        """Prepare subtitle entries for LLM analysis."""
        lines = []
        for i, entry in enumerate(subtitle_entries):
            timestamp = f"{entry.start_time:.1f}s-{entry.end_time:.1f}s"
            speaker = f"[{entry.speaker}] " if entry.speaker else ""
            lines.append(f"{i+1}. {timestamp}: {speaker}{entry.text}")
        
        return "\n".join(lines)
    
    def _create_filtering_prompt(
        self, 
        event: RecapEvent, 
        subtitle_text: str, 
        target_duration: float,
        context: Optional[str]
    ) -> str:
        """Create the LLM prompt for subtitle filtering."""
        
        context_section = f"\n\nEPISODE CONTEXT:\n{context}" if context else ""
        
        prompt = f"""You are an expert video editor creating a "Previously on..." recap for a TV show. Your task is to select the most compelling and contextually important subtitles from a scene to include in the recap.

EVENT CONTEXT:
- Content: {event.content}
- Series: {event.series} {event.season}{event.episode}
- Narrative Arc: {event.arc_title or 'Unknown'}
- Characters: {', '.join(event.main_characters) if event.main_characters else 'Unknown'}
- Target Duration: {target_duration} seconds{context_section}

AVAILABLE SUBTITLES:
{subtitle_text}

SELECTION CRITERIA:
1. **Dramatic Impact**: Prioritize emotionally compelling moments, tension, conflict, or revelation
2. **Narrative Importance**: Select lines that advance the story or reveal key information
3. **Character Development**: Include dialogue that shows character growth or relationships
4. **Clarity**: Choose lines that are understandable without extensive context
5. **Pacing**: Maintain good flow and rhythm for the recap
6. **Memorability**: Select iconic or quotable moments that viewers would remember

FILTERING GUIDELINES:
- Select 3-7 subtitle lines that best represent this event
- Prioritize quality over quantity - better to have fewer impactful lines
- Avoid purely expository or mundane dialogue
- Include emotional peaks and dramatic moments
- Consider both dialogue and any important action descriptions
- Ensure selected lines flow well together chronologically

Please analyze each subtitle and return a JSON response with your selections:

{{
    "selected_subtitles": [
        {{
            "subtitle_number": 1,
            "importance_score": 0.95,
            "reasoning": "Sets up the central conflict of the scene",
            "dramatic_value": "high",
            "content_type": "dialogue"
        }}
    ],
    "overall_analysis": "Brief explanation of your filtering strategy for this event",
    "estimated_duration": 8.5,
    "quality_assessment": "high"
}}

CONTENT TYPES:
- "dialogue": Character conversations
- "action": Action descriptions or sound effects
- "emotional": Expressions of emotion, crying, shouting
- "exposition": Information delivery or background

DRAMATIC VALUES:
- "high": Essential, emotional, or climactic moments
- "medium": Important but not critical
- "low": Background or transitional content

Respond only with valid JSON."""

        return prompt
    
    def _call_llm_for_filtering(self, prompt: str) -> Dict[str, Any]:
        """Call LLM for subtitle filtering analysis."""
        try:
            response = self.llm.invoke(prompt)
            
            # Extract content from the response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Try to parse as JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    return json.loads(response_text[json_start:json_end])
                else:
                    raise LLMServiceError("LLM response is not valid JSON")
                    
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            raise LLMServiceError(f"Failed to get LLM filtering response: {e}")
    
    def _parse_filtering_response(
        self, 
        llm_response: Dict[str, Any], 
        subtitle_entries: List[SubtitleEntry]
    ) -> List[FilteredSubtitle]:
        """Parse LLM response and create FilteredSubtitle objects."""
        try:
            selected_subtitles = []
            
            for selection in llm_response.get("selected_subtitles", []):
                subtitle_num = selection.get("subtitle_number", 1) - 1  # Convert to 0-based
                
                if 0 <= subtitle_num < len(subtitle_entries):
                    filtered_subtitle = FilteredSubtitle(
                        subtitle=subtitle_entries[subtitle_num],
                        importance_score=selection.get("importance_score", 0.5),
                        reasoning=selection.get("reasoning", ""),
                        dramatic_value=selection.get("dramatic_value", "medium"),
                        content_type=selection.get("content_type", "dialogue")
                    )
                    selected_subtitles.append(filtered_subtitle)
                else:
                    logger.warning(f"⚠️ Invalid subtitle number: {subtitle_num + 1}")
            
            # Sort by importance score
            selected_subtitles.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Log the analysis results
            logger.info(f"📊 LLM Analysis Results:")
            logger.info(f"   Overall Strategy: {llm_response.get('overall_analysis', 'N/A')}")
            logger.info(f"   Quality Assessment: {llm_response.get('quality_assessment', 'N/A')}")
            logger.info(f"   Estimated Duration: {llm_response.get('estimated_duration', 'N/A')}s")
            
            return selected_subtitles
            
        except Exception as e:
            logger.error(f"❌ Failed to parse LLM filtering response: {e}")
            # Fallback: return first few subtitles
            logger.warning("⚠️ Using fallback subtitle selection")
            return [
                FilteredSubtitle(
                    subtitle=entry,
                    importance_score=0.5,
                    reasoning="Fallback selection",
                    dramatic_value="medium",
                    content_type="dialogue"
                )
                for entry in subtitle_entries[:3]
            ]
    
    def filter_multiple_events(
        self, 
        events_with_subtitles: List[Tuple[RecapEvent, List[SubtitleEntry]]],
        global_context: Optional[str] = None
    ) -> Dict[str, List[FilteredSubtitle]]:
        """
        Filter subtitles for multiple events, considering global recap context.
        
        Args:
            events_with_subtitles: List of (event, subtitle_entries) tuples
            global_context: Overall context for the entire recap
            
        Returns:
            Dictionary mapping event IDs to filtered subtitles
        """
        try:
            logger.info(f"🎬 Filtering subtitles for {len(events_with_subtitles)} events")
            
            results = {}
            
            for event, subtitle_entries in events_with_subtitles:
                filtered_subtitles = self.filter_event_subtitles(
                    event=event,
                    subtitle_entries=subtitle_entries,
                    context=global_context
                )
                results[event.id] = filtered_subtitles
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Multiple event filtering failed: {e}")
            raise SubtitleProcessingError(f"Failed to filter multiple events: {e}")
    
    def optimize_subtitle_timing(
        self, 
        filtered_subtitles: List[FilteredSubtitle],
        target_duration: float
    ) -> List[FilteredSubtitle]:
        """
        Optimize subtitle timing to fit target duration while maintaining impact.
        
        Args:
            filtered_subtitles: List of filtered subtitles
            target_duration: Target duration in seconds
            
        Returns:
            Optimized list of subtitles
        """
        try:
            if not filtered_subtitles:
                return []
            
            # Calculate current duration
            current_duration = sum(
                subtitle.subtitle.end_time - subtitle.subtitle.start_time 
                for subtitle in filtered_subtitles
            )
            
            logger.info(f"⏱️ Optimizing subtitle timing: {current_duration:.1f}s → {target_duration:.1f}s")
            
            if current_duration <= target_duration:
                # Already within target, return as-is
                return filtered_subtitles
            
            # Need to reduce duration - prioritize by importance score
            optimized = []
            accumulated_duration = 0.0
            
            # Sort by importance (already sorted, but ensure)
            sorted_subtitles = sorted(
                filtered_subtitles, 
                key=lambda x: x.importance_score, 
                reverse=True
            )
            
            for subtitle in sorted_subtitles:
                subtitle_duration = subtitle.subtitle.end_time - subtitle.subtitle.start_time
                
                if accumulated_duration + subtitle_duration <= target_duration:
                    optimized.append(subtitle)
                    accumulated_duration += subtitle_duration
                else:
                    # Check if we can fit part of this subtitle
                    remaining_time = target_duration - accumulated_duration
                    if remaining_time > 2.0 and subtitle.importance_score > 0.8:
                        # Truncate high-importance subtitle if we have enough time
                        logger.info(f"✂️ Truncating subtitle for timing optimization")
                        optimized.append(subtitle)
                        break
                    else:
                        break
            
            logger.info(f"📝 Optimized to {len(optimized)} subtitles ({accumulated_duration:.1f}s)")
            return optimized
            
        except Exception as e:
            logger.error(f"❌ Subtitle timing optimization failed: {e}")
            return filtered_subtitles[:3]  # Fallback to first 3
    
    def analyze_subtitle_quality(
        self, 
        filtered_subtitles: List[FilteredSubtitle]
    ) -> Dict[str, Any]:
        """
        Analyze the quality of filtered subtitles.
        
        Args:
            filtered_subtitles: List of filtered subtitles to analyze
            
        Returns:
            Quality analysis metrics
        """
        try:
            if not filtered_subtitles:
                return {"quality_score": 0.0, "issues": ["No subtitles selected"]}
            
            # Calculate quality metrics
            avg_importance = sum(s.importance_score for s in filtered_subtitles) / len(filtered_subtitles)
            
            dramatic_distribution = {
                "high": sum(1 for s in filtered_subtitles if s.dramatic_value == "high"),
                "medium": sum(1 for s in filtered_subtitles if s.dramatic_value == "medium"),
                "low": sum(1 for s in filtered_subtitles if s.dramatic_value == "low")
            }
            
            content_distribution = {}
            for subtitle in filtered_subtitles:
                content_type = subtitle.content_type
                content_distribution[content_type] = content_distribution.get(content_type, 0) + 1
            
            # Quality assessment
            issues = []
            if avg_importance < 0.6:
                issues.append("Low average importance score")
            if dramatic_distribution["high"] == 0:
                issues.append("No high dramatic value subtitles")
            if len(set(s.content_type for s in filtered_subtitles)) == 1:
                issues.append("Limited content type diversity")
            
            quality_score = avg_importance
            if dramatic_distribution["high"] > 0:
                quality_score += 0.1
            if len(content_distribution) > 1:
                quality_score += 0.1
            
            quality_score = min(1.0, quality_score)
            
            return {
                "quality_score": quality_score,
                "average_importance": avg_importance,
                "dramatic_distribution": dramatic_distribution,
                "content_distribution": content_distribution,
                "total_subtitles": len(filtered_subtitles),
                "issues": issues,
                "recommendation": "Good quality selection" if quality_score > 0.7 else "Consider reviewing selection"
            }
            
        except Exception as e:
            logger.error(f"❌ Subtitle quality analysis failed: {e}")
            return {"quality_score": 0.0, "error": str(e)}
