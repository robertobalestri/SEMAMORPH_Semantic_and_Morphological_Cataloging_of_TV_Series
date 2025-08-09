#!/usr/bin/env python3
"""
Test script to validate round-robin fallback behavior in subtitle filtering.

This test ensures that when the primary event fails, the system correctly
tries the next event in the arc with the correct timespan.
"""

import os
import sys
import logging
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

def create_mock_subtitle_data_fallback() -> Dict[str, List[Dict]]:
    """Create mock subtitle data for testing round-robin fallback."""
    return {
        "GAS01E06": [
            # Event 1 timespan: 00:02:54 - 00:03:10 (ONLY 1 subtitle - not enough for consecutive)
            {"start": "00:02:54,538", "end": "00:03:10,145", "text": "Single subtitle - not enough for consecutive"},
            
            # Event 2 timespan: 00:12:12 - 00:12:25 (GOOD consecutive dialogue)
            {"start": "00:12:12,875", "end": "00:12:14,577", "text": "Dr. Meredith Gray: It's the chase, isn't it?"},
            {"start": "00:12:15,360", "end": "00:12:15,801", "text": "Dr. Derek Shepherd: What?"},
            {"start": "00:12:16,490", "end": "00:12:17,432", "text": "Dr. Meredith Gray: The thrill of the chase."},
            {"start": "00:12:18,133", "end": "00:12:20,057", "text": "Dr. Meredith Gray: I've been wondering about this."},
            
            # Event 3 timespan: 00:25:00 - 00:25:10 (Another good consecutive dialogue)  
            {"start": "00:25:00,000", "end": "00:25:02,000", "text": "Dr. Derek Shepherd: We need to talk."},
            {"start": "00:25:03,000", "end": "00:25:05,000", "text": "Dr. Meredith Gray: About what?"},
            {"start": "00:25:06,000", "end": "00:25:08,000", "text": "Dr. Derek Shepherd: About us."},
        ]
    }

def test_round_robin_fallback():
    """Test that round-robin fallback works correctly."""
    logger.info("🧪 Testing round-robin fallback behavior...")
    
    # Create mock events
    arc_id = "6ac77639-cc0d-49ed-a35b-81cfbfdb9991"
    arc_title = "Character Development Arc"
    
    # Event 1: Not enough subtitles (should fail)
    event1 = create_mock_event(
        event_id="event-1-insufficient",
        series="GA", season="S01", episode="E06",
        start_time="00:02:54,538", end_time="00:03:10,145",
        arc_id=arc_id, arc_title=arc_title
    )
    
    # Event 2: Good consecutive dialogue (should succeed as fallback)
    event2 = create_mock_event(
        event_id="event-2-good",
        series="GA", season="S01", episode="E06", 
        start_time="00:12:12,875", end_time="00:12:25,000",
        arc_id=arc_id, arc_title=arc_title
    )
    
    # Event 3: Another good dialogue (backup fallback)
    event3 = create_mock_event(
        event_id="event-3-backup",
        series="GA", season="S01", episode="E06",
        start_time="00:25:00,000", end_time="00:25:10,000", 
        arc_id=arc_id, arc_title=arc_title
    )
    
    events = [event1]  # We're trying to process event1
    subtitle_data = create_mock_subtitle_data_fallback()
    ranked_events_by_arc = {arc_id: [event1, event2, event3]}  # Fallback order
    
    # Test the extract_key_dialogue function with mocked LLM
    try:
        from recap_gen.llm_services import extract_key_dialogue
        from unittest.mock import patch, MagicMock
        
        # Mock LLM to return different responses based on input
        def mock_llm_responses(prompt):
            mock_response = MagicMock()
            if "00:02:54" in prompt:  # Event 1 - will fail due to insufficient subtitles
                mock_response.content = "SKIP"  # Not enough subtitles
            elif "00:12:12" in prompt:  # Event 2 - should succeed
                mock_response.content = "0,1,2,3"  # Select first 4 consecutive
            elif "00:25:00" in prompt:  # Event 3 - backup
                mock_response.content = "0,1,2"  # Select first 3 consecutive
            else:
                mock_response.content = "SKIP"
            return mock_response
        
        with patch('recap_gen.llm_services.get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = mock_llm_responses
            mock_get_llm.return_value = mock_llm
            
            # Call the function
            results = extract_key_dialogue(events, subtitle_data, ranked_events_by_arc, subtitle_data)
            
            # Validate results
            assert event1.id in results, f"Event {event1.id} not found in results"
            
            event_result = results[event1.id]
            logger.info(f"📊 Result for original event {event1.id}:")
            logger.info(f"   Lines: {event_result.get('lines', [])}")
            logger.info(f"   Start time: {event_result.get('start_time', '')}")
            logger.info(f"   End time: {event_result.get('end_time', '')}")
            logger.info(f"   Source event: {event_result.get('source_event_id', 'N/A')}")
            
            # Validate that we got dialogue from the fallback event (event2)
            if 'debug' in event_result:
                debug_info = event_result['debug']
                event_used = debug_info.get('event_used', '')
                status = debug_info.get('status', '')
                
                logger.info(f"   Debug info:")
                logger.info(f"     Event used: {event_used}")
                logger.info(f"     Status: {status}")
                
                # Check that fallback was used
                if event_used == event2.id:
                    logger.info("✅ Correctly used fallback event2 when event1 failed")
                    
                    # Check that dialogue is from the correct timespan (event2's timespan)
                    expected_dialogue = [
                        "Dr. Meredith Gray: It's the chase, isn't it?",
                        "Dr. Derek Shepherd: What?",
                        "Dr. Meredith Gray: The thrill of the chase.",
                        "Dr. Meredith Gray: I've been wondering about this."
                    ]
                    
                    actual_dialogue = event_result.get('lines', [])
                    
                    if actual_dialogue == expected_dialogue:
                        logger.info("✅ Extracted dialogue from correct fallback event timespan")
                        
                        # Check that timing is correct (should be from event2's timespan)
                        start_time = event_result.get('start_time', '')
                        if start_time.startswith("00:12:12"):
                            logger.info("✅ Start time matches fallback event timespan")
                            return True
                        else:
                            logger.error(f"❌ Wrong start time: {start_time}, expected 00:12:12,875")
                            return False
                    else:
                        logger.error(f"❌ Wrong dialogue content: {actual_dialogue}")
                        logger.error(f"   Expected: {expected_dialogue}")
                        return False
                else:
                    logger.error(f"❌ Wrong event used: {event_used}, expected {event2.id}")
                    return False
            else:
                logger.error("❌ No debug info found in results")
                return False
                
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting round-robin fallback test...")
    success = test_round_robin_fallback()
    
    if success:
        logger.info("🎉 Round-robin fallback test passed! Fallback mechanism working correctly.")
        sys.exit(0)
    else:
        logger.error("💥 Round-robin fallback test failed!")
        sys.exit(1)
