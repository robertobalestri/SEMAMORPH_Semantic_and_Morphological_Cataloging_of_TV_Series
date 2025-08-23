"""
LLM services for recap generation.

This module contains all three LLM calls in the recap generation workflow:
1. Query generation per narrative arc
2. Event ranking per arc  
3. Subtitle pruning per event
"""

import json
import logging
import re
import sys
import os
from typing import List, Dict, Any

# Add the src directory to Python path to enable proper imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ai_models.ai_models import get_llm, LLMType

logger = logging.getLogger(__name__)


def clean_llm_json_response(response: str) -> Dict:
    """Local implementation to avoid import issues."""
    try:
        # Remove from the string ```json, ```plaintext, ```markdown and ```
        response_cleaned = re.sub(r"```(json|plaintext|markdown)?", "", response)
        response_cleaned = response_cleaned.strip()
        
        # Remove any comments
        response_cleaned = re.sub(r'//.*?$|/\*.*?\*/', '', response_cleaned, flags=re.MULTILINE | re.DOTALL)
        
        # Try to extract a JSON object or array from the cleaned response
        json_match = re.search(r'(\{|\[)[\s\S]*(\}|\])', response_cleaned)
        
        if json_match:
            json_str = json_match.group(0)
            
            # Replace all curly apostrophes with regular apostrophes
            json_str = json_str.replace("'", "'")
            
            try:
                parsed_json = json.loads(json_str)
                
                # If the parsed JSON is a list, return first item or empty dict
                if isinstance(parsed_json, list):
                    return parsed_json[0] if parsed_json else {}
                elif isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.debug(f"JSON string that failed: {json_str}...")
        
        # If no JSON found, try to find just numbers (fallback)
        numbers = re.findall(r'\b\d+\b', response_cleaned)
        if numbers:
            logger.warning(f"No JSON found, extracted numbers: {numbers}")
            return {"selected_events": [int(x) for x in numbers]}
        
    except Exception as e:
        logger.error(f"Error in clean_llm_json_response: {e}")
    
    # Return empty dict if parsing fails
    logger.warning("Could not parse LLM response, returning empty dict")
    return {}


def generate_arc_queries(season_summary: str, episode_plot: str, narrative_arcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LLM #1: Generate vector database queries for each narrative arc.
    
    Args:
        season_summary: Season summary for context
        episode_plot: Current episode plot
        narrative_arcs: List of narrative arcs with title/description
        
    Returns:
        List of query dictionaries for vector database search
    """
    llm = get_llm(LLMType.INTELLIGENT)
    arc_queries = []
    
    for arc in narrative_arcs:
        prompt = f"""Generate 2-3 specific vector database search queries to find historical events that provide context for this narrative arc in the current episode.

**Season Context:** {season_summary}

**Current Episode Plot:** {episode_plot}

**Focus Narrative Arc:**
- Title: {arc['title']}
- Description: {arc['description']}

Generate queries that will find events showing:
1. Development and evolution of this arc
2. Character motivations within this arc  
3. Past conflicts/relationships that impact this arc

The most important thing is that the events proposed should be the most interesting and relevant to the current episode! We are creating a recap to help viewers understand the current episode.
Do not generate phrases such as "search for", "the scene where", "the moment when", "Key interactions where". These lines are all introductory but don't help the research. The query should be a direct narrative moment such as "John confronts Sarah about the missing files".

Output as JSON:
```json
{{
    "queries": [
        {{
            "query_text": "natural language search query",
            "purpose": "what context this provides"
        }}
    ]
}}
```"""

        try:
            response = llm.invoke(prompt)
            query_data = clean_llm_json_response(response.content)
            
            if isinstance(query_data, list) and query_data:
                query_data = query_data[0]
            
            queries = query_data.get('queries', [])
            
            for query in queries:
                arc_queries.append({
                    'query_text': query['query_text'],
                    'purpose': query.get('purpose', ''),
                    'narrative_arc_id': arc['narrative_arc_id'],
                    'arc_title': arc['title']
                })
                
        except Exception as e:
            logger.warning(f"Failed to generate queries for arc '{arc['title']}': {e}")
            # Fallback query
            arc_queries.append({
                'query_text': f"events related to {arc['title']}",
                'purpose': f"general context for {arc['title']}",
                'narrative_arc_id': arc['narrative_arc_id'],
                'arc_title': arc['title']
            })
    
    logger.info(f"Generated {len(arc_queries)} queries across {len(narrative_arcs)} arcs")
    return arc_queries


def rank_events_per_arc(events_by_arc: Dict[str, List[Any]], episode_plot: str) -> Dict[str, List[Any]]:
    """
    LLM #2: Select top 3 most important events per narrative arc.
    
    Args:
        events_by_arc: Dictionary mapping arc_id to list of events
        episode_plot: Current episode plot for context
        
    Returns:
        Dictionary mapping arc_id to ranked top 3 events
    """
    llm = get_llm(LLMType.INTELLIGENT)
    ranked_events = {}
    
    for arc_id, events in events_by_arc.items():
        if not events:
            continue
            
        # Limit to top 20 events to avoid token limits
        events = events
        
        event_summaries = []
        for i, event in enumerate(events):
            event_summaries.append(f"{i}: [{event.series}{event.season}E{event.episode}] {event.content}")
        
        prompt = f"""Select the 3 most ESSENTIAL events that provide crucial background for understanding the current episode.

**Current Episode Plot:** {episode_plot}...

**Historical Events:**
{chr(10).join(event_summaries)}

Select events that:
1. Directly explain character relationships/conflicts in current episode
2. Show crucial backstory for character motivations
3. Set up major plot developments that matter now

The most important thing is that the events selected should be the most interesting and relevant to the current episode! We are creating a recap to help viewers understand the current episode.

IMPORTANT: Respond with ONLY the event numbers, separated by commas. Example: "0, 5, 12" or "1, 3, 7"

Your answer (just the numbers):"""

        try:
            response = llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"LLM response for arc {arc_id}: {response_content}...")
            
            # Simple number parsing from comma-separated response
            response_clean = response_content.strip()
            
            # Try to extract numbers directly
            numbers = re.findall(r'\b\d+\b', response_clean)
            selected_indices = [int(x) for x in numbers]  # Take first 3 numbers
            
            # Convert indices to events
            selected_events = []
            for idx in selected_indices:
                if 0 <= idx < len(events):
                    selected_events.append(events[idx])
            
            if selected_events:
                ranked_events[arc_id] = selected_events
                logger.info(f"Arc {arc_id}: selected {len(selected_events)} events")
            else:
                logger.warning(f"Arc {arc_id}: LLM selection failed, using fallback")
                ranked_events[arc_id] = events # Fallback: all events
                logger.warning(f"Arc {arc_id}: LLM selection failed, using fallback")
                ranked_events[arc_id] = events
            
        except Exception as e:
            logger.warning(f"Failed to rank events for arc {arc_id}: {e}")
            # Fallback: take first 3 events
            ranked_events[arc_id] = events
    
    total_selected = sum(len(events) for events in ranked_events.values())
    logger.info(f"Ranked events across {len(ranked_events)} arcs: {total_selected} total events")
    
    return ranked_events


def extract_key_dialogue(events: List[Any], subtitle_data: Dict[str, List[Dict]], ranked_events_by_arc: Dict[str, List[Any]], all_subtitle_data: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
    """
    LLM #3: Extract most meaningful CONSECUTIVE dialogue lines from subtitle spans.
    
    This function implements a round-robin fallback mechanism with timestamp conflict resolution:
    1. For each selected event, try to extract dialogue from its original timespan
    2. If unsuccessful, try other events in the same arc, EXCLUDING already-used timestamp ranges
    3. Track used timestamp ranges globally to prevent duplicate video content
    4. If successful, record the dialogue with proper attribution to which event was actually used
    
    Args:
        events: List of selected events
        subtitle_data: Dictionary mapping episode to subtitle entries
        ranked_events_by_arc: Dictionary of ranked events by arc for fallback
        all_subtitle_data: All subtitle data for debug purposes
        
    Returns:
        Dictionary mapping event_id to dialogue extraction results
    """
    llm = get_llm(LLMType.INTELLIGENT)
    key_dialogue = {}
    
    logger.info(f"📝 Starting dialogue extraction for {len(events)} events")
    logger.info(f"📝 Event IDs: {[event.id[:8] + '...' for event in events]}")
    
    # Track which timestamp ranges are being used to detect conflicts
    used_timestamp_ranges = set()  # Format: "start_time-end_time"
    timestamp_to_event_map = {}    # Track which event used which timestamps
    
    for original_event in events:
        logger.info(f"--- Starting dialogue extraction for event {original_event.id} (Arc: {original_event.arc_title}) ---")
        success = False
        
        # Get all events in this arc for fallback
        arc_events = ranked_events_by_arc.get(original_event.narrative_arc_id, [original_event])
        
        # Smart sorting: prioritize events less likely to have timestamp conflicts
        def event_conflict_score(event):
            """Score events by likelihood of timestamp conflicts (lower is better)"""
            event_range = f"{event.start_time}-{event.end_time}"
            if event_range in used_timestamp_ranges:
                return 1000  # High penalty for exact range conflicts
            
            # Check for partial overlaps with used ranges
            event_start = _parse_timestamp_to_seconds(event.start_time)
            event_end = _parse_timestamp_to_seconds(event.end_time)
            overlap_penalty = 0
            
            for used_range in used_timestamp_ranges:
                if '-' in used_range:
                    used_start_str, used_end_str = used_range.split('-')
                    used_start = _parse_timestamp_to_seconds(used_start_str)
                    used_end = _parse_timestamp_to_seconds(used_end_str)
                    
                    # Check for overlap
                    if not (event_end < used_start or event_start > used_end):
                        overlap_penalty += 10  # Penalty for any overlap
            
            return overlap_penalty
        
        # Sort arc events by conflict likelihood (best candidates first)
        arc_events = sorted(arc_events, key=event_conflict_score)
        max_attempts = min(3, len(arc_events))  # Try up to 3 events or all available events
        
        last_attempt_inputs = {"subtitles": [], "llm_response": "", "status": "No attempts made"}

        for attempt in range(max_attempts):
            # Select which event to try (round-robin through arc events)
            if attempt < len(arc_events):
                current_event = arc_events[attempt]
            else:
                break  # No more events to try
                
            logger.info(f"Attempt {attempt + 1}/{max_attempts}: Trying event {current_event.id} for original event {original_event.id}")
            
            episode_key = f"{current_event.series}{current_event.season}{current_event.episode}"
            subtitles = subtitle_data.get(episode_key, [])
            
            if not subtitles:
                logger.warning(f"No subtitles found for {episode_key}")
                continue
                
            # Find subtitles in CURRENT event's timespan (not original event!)
            event_subtitles_with_timing = []
            start_seconds = _parse_timestamp_to_seconds(current_event.start_time)
            end_seconds = _parse_timestamp_to_seconds(current_event.end_time)
            
            # Pre-check: Does this event's timespan conflict with already-used ranges?
            current_event_range = f"{current_event.start_time}-{current_event.end_time}"
            if current_event_range in used_timestamp_ranges:
                logger.info(f"   ⚠️  Event {current_event.id[:8]}... timespan already used by {timestamp_to_event_map[current_event_range][:8]}..., skipping")
                continue
            
            for i, subtitle in enumerate(subtitles):
                sub_start = _parse_timestamp_to_seconds(subtitle['start'])
                sub_end = _parse_timestamp_to_seconds(subtitle['end'])
                
                # Check if subtitle starts within the CURRENT event timespan
                if sub_start >= start_seconds and sub_start <= end_seconds:
                    event_subtitles_with_timing.append({
                        'text': subtitle['text'],
                        'start': sub_start,
                        'end': sub_end,
                        'start_formatted': subtitle['start'],
                        'end_formatted': subtitle['end'],
                        'original_index': i
                    })
            
            if not event_subtitles_with_timing:
                logger.warning(f"No subtitles found in timespan for event {current_event.id} ({current_event.start_time} - {current_event.end_time})")
                continue
            
            # Sort by start time
            event_subtitles_with_timing.sort(key=lambda x: x['start'])
            
            if len(event_subtitles_with_timing) < 2:
                logger.warning(f"Not enough subtitles for consecutive selection in event {current_event.id} (found {len(event_subtitles_with_timing)})")
                continue

            # Present ALL subtitles to LLM for selection
            subtitle_list = []
            for j, subtitle in enumerate(event_subtitles_with_timing):
                subtitle_list.append(f"{j}: [{subtitle['start_formatted']} - {subtitle['end_formatted']}] {subtitle['text']}")
            
            logger.debug(f"Subtitles presented to LLM for event {current_event.id}:\n{chr(10).join(subtitle_list)}")

            prompt = f"""Select the BEST CONSECUTIVE dialogue sequence for a "Previously On" recap clip.

**Event Context:** {current_event.content}
**Arc:** {current_event.arc_title}
**Event Timespan:** {current_event.start_time} - {current_event.end_time}

**ALL Available Subtitles in Event Timespan:**
{chr(10).join(subtitle_list)}

INSTRUCTIONS:
1. Choose CONSECUTIVE subtitles that form a meaningful dialogue exchange
2. Maximum span: 10 seconds from first to last selected subtitle
3. Minimum: 2 consecutive subtitles
4. Maximum: As many consecutive subtitles as fit within 10 seconds
5. Choose dialogue that best represents this narrative moment

The most important thing is that the subtitles selected should be the most interesting and relevant to the current episode! We are creating a recap to help viewers understand the current episode.

OUTPUT FORMAT:
If good consecutive dialogue exists, respond with ONLY the subtitle numbers (e.g., "0,1,2" or "1,2,3,4")
If no good consecutive dialogue exists, respond just with "SKIP"

Your selection:"""

            last_attempt_inputs = {
                "subtitles": subtitle_list,
                "event_used": current_event.id,
                "event_timespan": f"{current_event.start_time} - {current_event.end_time}",
                "original_event": original_event.id if current_event.id != original_event.id else None
            }

            try:
                response = llm.invoke(prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)
                last_attempt_inputs["llm_response"] = response_content.strip()
                logger.debug(f"LLM raw response for event {current_event.id}: {response_content}")
                response_clean = response_content.strip()
                
                if response_clean.upper() == "SKIP":
                    logger.info(f"LLM decided to skip event {current_event.id}, trying next event")
                    last_attempt_inputs["status"] = "LLM chose to SKIP this event"
                    continue
                
                # Parse selected indices
                try:
                    selected_indices = [int(x.strip()) for x in response_clean.split(',')]
                    logger.debug(f"Successfully parsed indices: {selected_indices}")
                    
                    # Validate indices are consecutive and within range
                    selected_indices.sort()
                    if (len(selected_indices) >= 2 and 
                        all(selected_indices[i] + 1 == selected_indices[i+1] for i in range(len(selected_indices)-1)) and
                        all(0 <= idx < len(event_subtitles_with_timing) for idx in selected_indices)):
                        
                        # Get the selected subtitles (using the indices from our ordered list)
                        selected_subs = [event_subtitles_with_timing[idx] for idx in selected_indices]

                        # Check if selection fits within 10 seconds
                        first_sub = selected_subs[0]
                        last_sub = selected_subs[-1]
                        duration = last_sub['end'] - first_sub['start']
                        
                        if duration <= 10.0:
                            # Extract selected subtitles
                            selected_lines = [sub['text'] for sub in selected_subs]
                            
                            if selected_lines:
                                key_dialogue[original_event.id] = {
                                    'lines': selected_lines,
                                    'start_time': first_sub['start_formatted'],
                                    'end_time': last_sub['end_formatted'],
                                    'source_event_id': current_event.id,
                                    'debug': {
                                        "event_used": current_event.id,
                                        "event_timespan": f"{current_event.start_time} - {current_event.end_time}",
                                        "subtitles_sent_to_llm": subtitle_list,
                                        "llm_response": response_clean,
                                        "selected_indices": selected_indices,
                                        "duration_seconds": round(duration, 1),
                                        "status": f"SUCCESS - Used event {current_event.id} for original event {original_event.id}"
                                    }
                                }
                                
                                # Check for timestamp conflicts and track usage
                                timestamp_key = f"{first_sub['start_formatted']}-{last_sub['end_formatted']}"
                                if timestamp_key in used_timestamp_ranges:
                                    logger.warning(f"🚨 DIALOGUE CONFLICT: Event {original_event.id[:8]}... would use same dialogue timestamps as {timestamp_to_event_map[timestamp_key][:8]}...: {timestamp_key}")
                                    logger.warning(f"   Trying to find alternative dialogue in same event...")
                                    last_attempt_inputs["status"] = f"Dialogue conflict with {timestamp_to_event_map[timestamp_key][:8]}..., trying alternative"
                                    
                                    # Try to find alternative consecutive dialogue in the same event
                                    alternative_found = False
                                    for alt_start_idx in range(len(event_subtitles_with_timing) - 1):
                                        if alt_start_idx in selected_indices:
                                            continue  # Skip already selected indices
                                        
                                        # Try different consecutive segments
                                        for alt_length in range(2, min(5, len(event_subtitles_with_timing) - alt_start_idx + 1)):
                                            alt_indices = list(range(alt_start_idx, alt_start_idx + alt_length))
                                            alt_subs = [event_subtitles_with_timing[idx] for idx in alt_indices]
                                            alt_duration = alt_subs[-1]['end'] - alt_subs[0]['start']
                                            alt_timestamp_key = f"{alt_subs[0]['start_formatted']}-{alt_subs[-1]['end_formatted']}"
                                            
                                            if alt_duration <= 10.0 and alt_timestamp_key not in used_timestamp_ranges:
                                                # Found alternative dialogue!
                                                selected_indices = alt_indices
                                                selected_subs = alt_subs
                                                first_sub = alt_subs[0]
                                                last_sub = alt_subs[-1]
                                                duration = alt_duration
                                                timestamp_key = alt_timestamp_key
                                                alternative_found = True
                                                logger.info(f"   ✅ Found alternative dialogue: {timestamp_key} ({duration:.1f}s)")
                                                break
                                        
                                        if alternative_found:
                                            break
                                    
                                    if not alternative_found:
                                        logger.warning(f"   ❌ No alternative dialogue found in event {current_event.id[:8]}..., trying next event")
                                        continue  # Try next event in arc
                                
                                # Record timestamp usage
                                used_timestamp_ranges.add(timestamp_key)
                                timestamp_to_event_map[timestamp_key] = original_event.id
                                
                                logger.info(f"✅ Event {original_event.id[:8]}... -> {len(selected_lines)} dialogue lines from {current_event.id[:8]}... at {first_sub['start_formatted']}-{last_sub['end_formatted']} ({duration:.1f}s)")
                                if current_event.id != original_event.id:
                                    logger.info(f"   🔄 Used different source event: {original_event.id[:8]}... -> {current_event.id[:8]}...")
                                success = True
                                break
                        else:
                            last_attempt_inputs["status"] = f"Selected dialogue span too long ({duration:.1f}s > 10s)"
                            logger.warning(f"Selected dialogue span for event {current_event.id} too long ({duration:.1f}s > 10s), trying next event")
                    else:
                        last_attempt_inputs["status"] = "Invalid selection (not consecutive or out of range)"
                        logger.warning(f"Invalid selection (not consecutive or out of range), trying next event")
                        
                except (ValueError, IndexError) as e:
                    last_attempt_inputs["status"] = f"Could not parse LLM selection: {e}"
                    logger.warning(f"Could not parse LLM selection '{response_clean}': {e}")
                
            except Exception as e:
                last_attempt_inputs["status"] = f"LLM call failed: {e}"
                logger.warning(f"Failed to extract dialogue for event {current_event.id}: {e}")
        
        # If no success after trying all events in the arc
        if not success:
            # Get the original event's subtitle content for debug
            episode_key = f"{original_event.series}{original_event.season}{original_event.episode}"
            subtitles_in_original_event = []
            if episode_key in all_subtitle_data:
                start_seconds = _parse_timestamp_to_seconds(original_event.start_time)
                end_seconds = _parse_timestamp_to_seconds(original_event.end_time)
                for sub in all_subtitle_data[episode_key]:
                    sub_start = _parse_timestamp_to_seconds(sub['start'])
                    if sub_start >= start_seconds and sub_start <= end_seconds:
                        subtitles_in_original_event.append(sub['text'])
            
            key_dialogue[original_event.id] = {
                "lines": [],
                "start_time": "",
                "end_time": "",
                "debug": {
                    **last_attempt_inputs,
                    "subtitles_in_original_event": subtitles_in_original_event,
                    "original_event_timespan": f"{original_event.start_time} - {original_event.end_time}",
                    "status": f"FAILED after {max_attempts} attempts"
                }
            }
            logger.warning(f"❌ Event {original_event.id}: failed to extract dialogue after trying {max_attempts} events in arc")

    logger.info(f"Extracted dialogue for {len([k for k, v in key_dialogue.items() if v['lines']])} out of {len(events)} events")
    return key_dialogue
    


def _parse_timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS,mmm timestamp to seconds."""
    try:
        if ',' in timestamp:
            time_part, ms_part = timestamp.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            return h * 3600 + m * 60 + s + ms / 1000.0
        else:
            h, m, s = map(int, timestamp.split(':'))
            return h * 3600 + m * 60 + s
    except:
        return 0.0
