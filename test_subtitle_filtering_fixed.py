#!/usr/bin/env python3
"""
Test script to validate that subtitle filtering is working correctly after the fix.

This script creates mock data to reproduce the bug and verify the fix:
1. Creates mock events with specific timespans 
2. Creates mock subtitle data with subtitles in different timespans
3. Calls extract_key_dialogue and validates results
4. Ensures subtitles sent to LLM match the event timespan
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any

# Add the backend/src directory to Python path
backend_src = os.path.join(os.path.dirname(__file__), 'backend', 'src')
sys.path.insert(0, backend_src)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_event(event_id: str, series: str, season: str, episode: str, 
                      start_time: str, end_time: str, arc_id: str, arc_title: str) -> Any:
    """Create a mock Event object."""
    from recap_gen.models import Event
    return Event(
        id=event_id,
        content=f"Mock event content for {arc_title}",
        series=series,
        season=season,
        episode=episode,
        start_time=start_time,
        end_time=end_time,
        narrative_arc_id=arc_id,
        arc_title=arc_title,
        relevance_score=1.0
    )

def create_mock_subtitle_data() -> Dict[str, List[Dict]]:
    """Create mock subtitle data that reproduces the bug scenario."""
    return {
        "GAS01E06": [
            # Subtitles for the CORRECT timespan (00:02:54 - 00:03:10)
            {"start": "00:02:54,538", "end": "00:02:56,000", "text": "Dr. Derek Shepherd: If I was a better guy, I'd walk away."},
            {"start": "00:02:57,000", "end": "00:02:58,500", "text": "Dr. Meredith Gray: Yes, you would."},
            {"start": "00:02:59,000", "end": "00:03:01,000", "text": "Dr. Derek Shepherd: Do you want me to be a better guy?"},
            {"start": "00:03:02,000", "end": "00:03:03,000", "text": "Dr. Meredith Gray: Yes."},
            {"start": "00:03:04,000", "end": "00:03:05,000", "text": "Dr. Meredith Gray: No."},
            {"start": "00:03:06,000", "end": "00:03:10,145", "text": "Dr. Meredith Gray: Crap."},
            
            # Subtitles in a DIFFERENT timespan (00:12:12 - 00:12:37) - these should NOT be selected
            {"start": "00:12:12,875", "end": "00:12:14,577", "text": "Intern / Dr. Meredith Gray (PVM): It's the chase, isn't it?"},
            {"start": "00:12:15,360", "end": "00:12:15,801", "text": "Supervisor/Another Doctor / Dr. Meredith Gray (PVM): What?"},
            {"start": "00:12:16,490", "end": "00:12:17,432", "text": "Intern / Dr. Meredith Gray (PVM): The thrill of the chase."},
            {"start": "00:12:18,133", "end": "00:12:22,057", "text": "Intern / Dr. Meredith Gray (PVM): I've been wondering to myself, why are you so hell-bent on getting me to go out with you?"},
            {"start": "00:12:22,600", "end": "00:12:23,441", "text": "Intern / Dr. Meredith Gray (PVM): You know you're my boss."},
            {"start": "00:12:23,760", "end": "00:12:24,783", "text": "Intern / Dr. Meredith Gray (PVM): You know it's against the rules."},
            {"start": "00:12:25,264", "end": "00:12:26,326", "text": "Intern / Dr. Meredith Gray (PVM): You know I keep saying no."},
            {"start": "00:12:27,187", "end": "00:12:27,726", "text": "Supervisor/Another Doctor / Dr. Meredith Gray (PVM): It's the chase."},
            {"start": "00:12:29,149", "end": "00:12:30,072", "text": "Supervisor/Another Doctor / Dr. Derek Shepherd (PVM): Well, it's fun, isn't it?"},
            {"start": "00:12:30,471", "end": "00:12:30,812", "text": "Intern / Dr. Meredith Gray (PVM): You see?"},
            {"start": "00:12:31,533", "end": "00:12:33,697", "text": "Intern / Dr. Meredith Gray (PVM): This is a game to you, but not to me."},
            {"start": "00:12:34,778", "end": "00:12:37,384", "text": "Character A / Dr. Meredith Gray (PVM): Because unlike you, I still have something to prove."}
        ]
    }

def test_subtitle_filtering():
    """Test the fixed subtitle filtering functionality."""
    logger.info("🧪 Testing subtitle filtering after fix...")
    
    # Create mock data
    arc_id = "6ac77639-cc0d-49ed-a35b-81cfbfdb9991"
    arc_title = "Meredith Grey And Derek Shepherd: Secret Romance And Professional Boundaries"
    
    # Create the problematic event from the bug report
    event = create_mock_event(
        event_id="110b80e7-a6d2-4c1a-8d10-b016b62bcdbe",
        series="GA",
        season="S01",
        episode="E06",
        start_time="00:02:54,538",
        end_time="00:03:10,145",
        arc_id=arc_id,
        arc_title=arc_title
    )
    
    # Create a second event in the same arc (for fallback testing)
    event2 = create_mock_event(
        event_id="222b80e7-a6d2-4c1a-8d10-b016b62bcdbe",
        series="GA",
        season="S01",
        episode="E06",
        start_time="00:12:12,875",
        end_time="00:12:37,384",
        arc_id=arc_id,
        arc_title=arc_title
    )
    
    events = [event]
    subtitle_data = create_mock_subtitle_data()
    ranked_events_by_arc = {arc_id: [event, event2]}  # Event is primary, event2 is fallback
    
    # Test the extract_key_dialogue function
    try:
        from recap_gen.llm_services import extract_key_dialogue
        
        # Mock the LLM to return a valid selection
        from unittest.mock import patch, MagicMock
        
        mock_response = MagicMock()
        mock_response.content = "0,1,2,3"  # Select first 4 consecutive subtitles
        
        with patch('recap_gen.llm_services.get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            # Call the function
            results = extract_key_dialogue(events, subtitle_data, ranked_events_by_arc, subtitle_data)
            
            # Validate results
            assert event.id in results, f"Event {event.id} not found in results"
            
            event_result = results[event.id]
            logger.info(f"📊 Result for event {event.id}:")
            logger.info(f"   Lines: {event_result.get('lines', [])}")
            logger.info(f"   Start time: {event_result.get('start_time', '')}")
            logger.info(f"   End time: {event_result.get('end_time', '')}")
            
            if 'debug' in event_result:
                debug_info = event_result['debug']
                logger.info(f"   Debug info:")
                logger.info(f"     Event used: {debug_info.get('event_used', 'N/A')}")
                logger.info(f"     Event timespan: {debug_info.get('event_timespan', 'N/A')}")
                logger.info(f"     Status: {debug_info.get('status', 'N/A')}")
                
                # Check if subtitles sent to LLM are from the correct timespan
                if 'subtitles_sent_to_llm' in debug_info:
                    sent_subtitles = debug_info['subtitles_sent_to_llm']
                    logger.info(f"   Subtitles sent to LLM ({len(sent_subtitles)} total):")
                    for i, subtitle in enumerate(sent_subtitles[:3]):  # Show first 3
                        logger.info(f"     {i}: {subtitle}")
                    if len(sent_subtitles) > 3:
                        logger.info(f"     ... and {len(sent_subtitles) - 3} more")
                    
                    # VALIDATION: Check that ALL subtitles sent to LLM are within the event timespan
                    event_start_seconds = parse_timestamp_to_seconds(event.start_time)
                    event_end_seconds = parse_timestamp_to_seconds(event.end_time)
                    
                    all_in_timespan = True
                    for subtitle_line in sent_subtitles:
                        # Extract timestamp from subtitle line format: "0: [00:02:54,538 - 00:02:56,000] Text"
                        if '[' in subtitle_line and ']' in subtitle_line:
                            timestamp_part = subtitle_line.split('[')[1].split(']')[0]
                            if ' - ' in timestamp_part:
                                start_time_str = timestamp_part.split(' - ')[0]
                                start_seconds = parse_timestamp_to_seconds(start_time_str)
                                
                                if not (event_start_seconds <= start_seconds <= event_end_seconds):
                                    all_in_timespan = False
                                    logger.error(f"❌ Subtitle outside event timespan: {subtitle_line}")
                                    logger.error(f"   Event timespan: {event_start_seconds}s - {event_end_seconds}s")
                                    logger.error(f"   Subtitle start: {start_seconds}s")
                    
                    if all_in_timespan:
                        logger.info("✅ All subtitles sent to LLM are within the correct event timespan!")
                    else:
                        logger.error("❌ Some subtitles sent to LLM are outside the event timespan - BUG NOT FIXED!")
                        return False
            
            # Check that we got the expected dialogue lines
            expected_dialogue = [
                "Dr. Derek Shepherd: If I was a better guy, I'd walk away.",
                "Dr. Meredith Gray: Yes, you would.",
                "Dr. Derek Shepherd: Do you want me to be a better guy?",
                "Dr. Meredith Gray: Yes."
            ]
            
            actual_dialogue = event_result.get('lines', [])
            
            if actual_dialogue == expected_dialogue:
                logger.info("✅ Extracted dialogue matches expected content!")
            else:
                logger.warning("⚠️ Extracted dialogue differs from expected:")
                logger.warning(f"   Expected: {expected_dialogue}")
                logger.warning(f"   Actual: {actual_dialogue}")
            
            # Verify that we did NOT get dialogue from the wrong timespan (00:12:12 area)
            wrong_dialogue = ["It's the chase, isn't it?", "What?", "The thrill of the chase."]
            dialogue_text = ' '.join(actual_dialogue)
            
            contains_wrong_dialogue = any(wrong_line in dialogue_text for wrong_line in wrong_dialogue)
            if not contains_wrong_dialogue:
                logger.info("✅ Did not extract dialogue from wrong timespan - Bug is fixed!")
                return True
            else:
                logger.error("❌ Still extracting dialogue from wrong timespan - Bug NOT fixed!")
                return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_timestamp_to_seconds(timestamp: str) -> float:
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

if __name__ == "__main__":
    logger.info("🚀 Starting subtitle filtering test...")
    success = test_subtitle_filtering()
    
    if success:
        logger.info("🎉 All tests passed! Subtitle filtering fix is working correctly.")
        sys.exit(0)
    else:
        logger.error("💥 Tests failed! There are still issues with subtitle filtering.")
        sys.exit(1)
