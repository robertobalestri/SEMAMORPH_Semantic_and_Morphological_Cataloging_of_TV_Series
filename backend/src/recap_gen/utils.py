"""
Utility functions for recap generation.
"""

import os
import json
import sqlite3
import logging
import sys
from typing import List, Dict, Any, Optional

# Add the src directory to Python path to enable proper imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Add the backend/src directory to sys.path for absolute imports
backend_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_src not in sys.path:
    sys.path.insert(0, backend_src)

from narrative_storage_management.vector_store_service import VectorStoreService
from path_handler import PathHandler

logger = logging.getLogger(__name__)


def load_episode_inputs(series: str, season: str, episode: str, base_dir: str = "data") -> Dict[str, Any]:
    """
    Load all required input files for recap generation.
    
    Args:
        series: Series identifier
        season: Season identifier 
        episode: Episode identifier
        base_dir: Base data directory
        
    Returns:
        Dictionary containing episode inputs
    """
    path_handler = PathHandler(series, season, episode, base_dir)
    
    inputs = {
        'series': series,
        'season': season, 
        'episode': episode
    }
    
    # Load episode plot
    plot_file = path_handler.get_plot_possible_speakers_path()
    if os.path.exists(plot_file):
        with open(plot_file, 'r', encoding='utf-8') as f:
            inputs['episode_plot'] = f.read().strip()
    else:
        logger.warning(f"Plot file not found: {plot_file}")
        inputs['episode_plot'] = ""
    
    # Load season summary
    summary_file = path_handler.get_season_summary_path()
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            inputs['season_summary'] = f.read().strip()
    else:
        logger.warning(f"Season summary not found: {summary_file}")
        inputs['season_summary'] = ""
    
    # Load running plotlines
    plotlines_file = path_handler.get_present_running_plotlines_path()
    if os.path.exists(plotlines_file):
        with open(plotlines_file, 'r', encoding='utf-8') as f:
            plotlines_data = json.load(f)
            
        # Handle different formats
        if isinstance(plotlines_data, list):
            narrative_arcs = plotlines_data
        else:
            narrative_arcs = plotlines_data.get('running_plotlines', [])
        
        # Enrich with narrative arc IDs
        for arc in narrative_arcs:
            arc_id = get_narrative_arc_id(arc.get('title', ''))
            arc['narrative_arc_id'] = arc_id
            
        inputs['narrative_arcs'] = narrative_arcs
    else:
        logger.warning(f"Plotlines file not found: {plotlines_file}")
        inputs['narrative_arcs'] = []
    
    # Load subtitle data
    inputs['subtitle_data'] = load_subtitle_data(path_handler)
    
    logger.info(f"Loaded inputs for {series}{season}{episode}: "
               f"{len(inputs['narrative_arcs'])} arcs, "
               f"{len(inputs.get('subtitle_data', {}))} subtitle files")
    
    return inputs


def get_narrative_arc_id(title: str) -> Optional[str]:
    """Get narrative arc ID from database by title."""
    try:
        db_path = "narrative_storage/narrative.db"
        if not os.path.exists(db_path):
            return None
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM narrativearc WHERE title = ?", (title,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
        
    except Exception as e:
        logger.warning(f"Failed to lookup arc ID for '{title}': {e}")
        return None


def load_subtitle_data(path_handler: PathHandler) -> Dict[str, List[Dict]]:
    """
    Load subtitle data for episodes that might be referenced.
    
    Uses PathHandler to get the correct subtitle files (prioritizing possible_speakers.srt).
    """
    subtitle_data = {}
    
    # Get season directory from path handler
    series = path_handler.get_series()
    season = path_handler.get_season()
    base_dir = path_handler.base_dir
    season_dir = f"{base_dir}/{series}/{season}"
    
    if not os.path.exists(season_dir):
        return subtitle_data
    
    # Load subtitles for all episodes in season
    for ep_dir in os.listdir(season_dir):
        ep_path = os.path.join(season_dir, ep_dir)
        if not os.path.isdir(ep_path):
            continue
        
        try:
            # Extract episode info (e.g., "E01" from "E01")
            if len(ep_dir) >= 3 and ep_dir.startswith('E'):
                episode = ep_dir  # ep_dir is already "E01", "E02", etc.
            else:
                continue
            
            # Create PathHandler for this specific episode
            episode_path_handler = PathHandler(series, season, episode, base_dir)
            
            # Create proper episode key format (e.g., "GAS01E01")
            episode_key = f"{series}{season}{episode}"  # Build the full key
            
            # Try to load possible_speakers.srt first (preferred)
            possible_speakers_path = episode_path_handler.get_possible_speakers_srt_path()
            if os.path.exists(possible_speakers_path):
                subtitle_entries = parse_srt_file(possible_speakers_path)
                subtitle_data[episode_key] = subtitle_entries
                logger.debug(f"Loaded possible_speakers subtitles for {episode_key}: {len(subtitle_entries)} entries")
                continue
            
            # Fallback to regular .srt file
            regular_srt_path = episode_path_handler.get_srt_file_path()
            if os.path.exists(regular_srt_path):
                subtitle_entries = parse_srt_file(regular_srt_path)
                subtitle_data[episode_key] = subtitle_entries
                logger.debug(f"Loaded regular subtitles for {episode_key}: {len(subtitle_entries)} entries")
                continue
                
            logger.debug(f"No subtitle files found for episode {episode_key}")
            
        except Exception as e:
            logger.warning(f"Failed to load subtitles for {ep_dir}: {e}")
            continue
    
    logger.info(f"Loaded subtitles for {len(subtitle_data)} episodes")
    return subtitle_data


def parse_srt_file(srt_path: str) -> List[Dict]:
    """
    Parse an SRT subtitle file.
    
    Returns:
        List of subtitle entries with start/end times and text
    """
    entries = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        blocks = content.split('\n\n')  # Fixed: use actual newlines, not literal \\n
        
        for block in blocks:
            lines = block.split('\n')  # Fixed: use actual newlines, not literal \\n
            if len(lines) >= 3:
                # Skip sequence number (first line)
                timestamp_line = lines[1]
                text_lines = lines[2:]
                
                # Parse timestamp: 00:00:20,000 --> 00:00:24,400
                if ' --> ' in timestamp_line:
                    start_time, end_time = timestamp_line.split(' --> ')
                    
                    entries.append({
                        'start': start_time.strip(),
                        'end': end_time.strip(),
                        'text': ' '.join(text_lines).strip()
                    })
    
    except Exception as e:
        logger.warning(f"Failed to parse SRT file {srt_path}: {e}")
    
    return entries


def select_events_round_robin(ranked_events_by_arc: Dict[str, List[Any]], max_events: int = 8) -> List[Any]:
    """
    Select final events using round-robin approach to ensure arc coverage.
    
    Args:
        ranked_events_by_arc: Dict mapping arc_id to ranked events
        max_events: Maximum number of events to select
        
    Returns:
        List of selected events
    """
    selected_events = []
    arc_ids = list(ranked_events_by_arc.keys())
    
    if not arc_ids:
        return selected_events
    
    # Round 1: Select one event from each arc (guaranteed coverage)
    for arc_id in arc_ids:
        events = ranked_events_by_arc[arc_id]
        if events:
            selected_events.append(events[0])
    
    # Round 2-3: Add additional events while under limit
    round_num = 1
    while len(selected_events) < max_events and round_num < 3:
        added_this_round = 0
        
        for arc_id in arc_ids:
            if len(selected_events) >= max_events:
                break
                
            events = ranked_events_by_arc[arc_id]
            if len(events) > round_num:
                selected_events.append(events[round_num])
                added_this_round += 1
        
        if added_this_round == 0:
            break
            
        round_num += 1
    
    logger.info(f"Round-robin selection: {len(selected_events)} events from {len(arc_ids)} arcs")
    return selected_events


def search_vector_database(queries: List[Dict[str, Any]], current_series: str = None, current_season: str = None, current_episode: str = None) -> Dict[str, List[Any]]:
    """
    Search vector database for events matching the queries.
    
    Args:
        queries: List of query dictionaries
        current_series: Current series for exclusion list
        current_season: Current season for exclusion list  
        current_episode: Current episode for exclusion list
        
    Returns:
        Dictionary mapping arc_id to list of matching events
    """
    try:
        from .models import Event
        
        # Initialize vector service using the same approach as working system
        vector_service = VectorStoreService()
        events_by_arc = {}
        
        # Build exclusion list (exclude current and future episodes)
        if current_season and current_episode:
            exclude_episodes = build_exclusion_list(current_season, current_episode)
        elif queries:
            exclude_episodes = build_exclusion_list('S01', 'E09')  # Default for testing
        else:
            exclude_episodes = []
        
        logger.info(f"🔍 Searching vector database for {len(queries)} queries")
        logger.info(f"🚫 Excluding {len(exclude_episodes)} future episodes from search")
        
        for query in queries:
            try:
                logger.debug(f"Executing query for arc {query['narrative_arc_id'][:8]}: '{query['query_text']}'")
                # Search vector database using SAME method as original
                results = vector_service.find_similar_events(
                    query['query_text'],
                    n_results=4,  # Same as original
                    series=current_series or 'GA',
                    narrative_arc_ids=[query['narrative_arc_id']] if query.get('narrative_arc_id') else None,
                    exclude_episodes=exclude_episodes
                )
                
                # Convert results to Event objects
                arc_id = query['narrative_arc_id']
                if arc_id not in events_by_arc:
                    events_by_arc[arc_id] = []
                
                for result in results:
                    # Extract metadata from vector search result
                    metadata = result.get('metadata', {})
                    
                    # Create Event object with proper data from vector search
                    event = Event(
                        id=metadata.get('id', ''),
                        content=result.get('page_content', ''),
                        series=metadata.get('series', ''),
                        season=metadata.get('season', ''),
                        episode=metadata.get('episode', ''),
                        start_time=metadata.get('start_timestamp', '00:00:00,000'),
                        end_time=metadata.get('end_timestamp', '00:00:00,000'),
                        narrative_arc_id=arc_id,
                        arc_title=query.get('arc_title', ''),
                        relevance_score=1.0 - result.get('cosine_distance', 1.0)  # Convert distance to similarity
                    )
                    events_by_arc[arc_id].append(event)
                    
                logger.debug(f"🎯 Query for arc {arc_id[:8]}... found {len(results)} events")
                
            except Exception as e:
                logger.warning(f"Vector search failed for query {query.get('query_text', '')[:30]}...: {e}")
                continue
        
        total_events = sum(len(events) for events in events_by_arc.values())
        logger.info(f"✅ Vector search completed: {total_events} events across {len(events_by_arc)} arcs")
        
        return events_by_arc
        
    except Exception as e:
        logger.error(f"❌ Vector database search failed: {e}")
        import traceback
        traceback.print_exc()
        raise e  # Don't return empty dict, raise the error so we know what's wrong


def build_exclusion_list(current_season: str, current_episode: str) -> List[tuple]:
    """Build list of episodes to exclude (current + future)."""
    exclude_episodes = []
    
    try:
        season_num = int(current_season[1:])  # S01 -> 1
        episode_num = int(current_episode[1:])  # E09 -> 9
    except:
        return [(current_season, current_episode)]
    
    # Exclude current episode
    exclude_episodes.append((current_season, current_episode))
    
    # Exclude future episodes in current season
    for ep in range(episode_num + 1, 25):
        exclude_episodes.append((current_season, f"E{ep:02d}"))
    
    # Exclude future seasons  
    for season in range(season_num + 1, 21):
        for ep in range(1, 25):
            exclude_episodes.append((f"S{season:02d}", f"E{ep:02d}"))
    
    return exclude_episodes
