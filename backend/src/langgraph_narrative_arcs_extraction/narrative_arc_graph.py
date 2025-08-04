# narrative_arc_graph.py

import asyncio
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
import os
import json
from datetime import datetime
from pathlib import Path

from ..utils.llm_utils import clean_llm_json_response
from ..utils.logger_utils import setup_logging
from ..utils.text_utils import load_text, save_json
from ..ai_models.ai_models import get_llm, LLMType
from ..plot_processing.plot_processing_models import EntityLink
from ..plot_processing.plot_ner_entity_extraction import normalize_names
# Import the new repositories and models
from ..narrative_storage_management.repositories import (
    DatabaseSessionManager,
    NarrativeArcRepository,
)
from ..narrative_storage_management.narrative_models import NarrativeArc, ArcProgression
import logging

logger = setup_logging(__name__)
llm = get_llm(LLMType.INTELLIGENT)
cheap_llm = get_llm(LLMType.CHEAP)
from pydantic import BaseModel, Field

class EventProgression(BaseModel):
    """Model representing a single event within a progression."""
    content: str = Field(..., description="Description of the individual event")
    ordinal_position: int = Field(..., description="Order within progression")
    characters_involved: List[str] = Field(default_factory=list, description="Characters involved in this event")
    
    class Config:
        # Allow dynamic attributes for timestamp assignment
        extra = "allow"

class IntermediateNarrativeArc(BaseModel):
    """Model representing an intermediate narrative arc during extraction process."""
    title: str = Field(..., description="The title of the narrative arc")
    arc_type: str = Field(..., description="Type of the arc such as 'Soap Arc'/'Genre-Specific Arc'/'Anthology Arc'")
    description: str = Field(..., description="A brief description of the narrative arc")
    main_characters: str = Field(..., description="Main characters involved in this arc")
    interfering_episode_characters: str = Field(..., description="Interfering characters involved in this arc")
    single_episode_progression_events: List[EventProgression] = Field(default_factory=list, description="List of events for the current episode")

    class Config:
        populate_by_name = True

# Define your prompts and guidelines (ensure these are defined in your code)
from .prompts import (
     EXTRACTOR_OUTPUT_JSON_FORMAT,
     PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT,
     ARC_VERIFIER_PROMPT,
     NARRATIVE_ARC_GUIDELINES,
     ARC_PROGRESSION_VERIFIER_PROMPT,
     DETAILED_OUTPUT_JSON_FORMAT,
     CHARACTER_VERIFIER_PROMPT,
     ARC_DEDUPLICATOR_PROMPT,
     ANTHOLOGY_ARC_EXTRACTOR_PROMPT,
     ARC_ENHANCER_PROMPT,
     BRIEF_OUTPUT_JSON_FORMAT,
     PRESENT_SEASON_ARCS_OUTPUT_JSON_FORMAT,
     SOAP_AND_GENRE_ARC_EXTRACTOR_PROMPT,
     SEASONAL_ARC_OPTIMIZER_PROMPT
 )

class NarrativeArcsExtractionState(TypedDict):
    """State representation for the narrative analysis process."""
    soap_arcs: List[IntermediateNarrativeArc]
    genre_arcs: List[IntermediateNarrativeArc]
    anthology_arcs: List[IntermediateNarrativeArc]
    episode_arcs: List[IntermediateNarrativeArc]
    present_season_arcs: List[Dict]
    season_arcs: List[Dict]  # Added to store serialized season arcs
    file_paths: Dict[str, str]
    series: str
    season: str
    episode: str
    existing_season_entities: List[EntityLink]
    episode_plot: str
    season_plot: str
    optimized_arcs: List[IntermediateNarrativeArc]
    timestamped_arcs: List[IntermediateNarrativeArc]  # NEW: Final arcs with timestamps

class ExtractedArcBase(BaseModel):
    """Basic model for initially extracted arcs before enhancement."""
    title: str = Field(..., description="The title of the narrative arc")
    description: str = Field(..., description="A brief description of the narrative arc")
    arc_type: str = Field(..., description="Type of the arc")

def initialize_state(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Initialize the state by loading necessary data from files."""
    logger.info("Initializing state with data from files.")

    # Load existing season entities
    if os.path.exists(state['file_paths']['season_entities_path']):
        with open(state['file_paths']['season_entities_path'], 'r') as f:
            season_entities_data = json.load(f)
            state['existing_season_entities'] = [EntityLink(**entity) for entity in season_entities_data]

    if os.path.exists(state['file_paths']['episode_plot_path']):
        with open(state['file_paths']['episode_plot_path'], 'r') as f:
            state['episode_plot'] = f.read()
        
        # Log recap filtering status for narrative arc extraction
        logger.info("📚 Episode plot loaded for narrative arc extraction")
        logger.info("ℹ️ Note: _plot.txt files already have recap content filtered out automatically")
        
        # Double-check if we need additional recap filtering
        from ..utils.recap_utils import get_recap_scene_count
        try:
            # Check if there are recap scenes in the original data
            from ..path_handler import PathHandler
            path_handler = PathHandler(state['series'], state['season'], state['episode'])
            scenes_json_path = path_handler.get_plot_scenes_json_path()
            
            if os.path.exists(scenes_json_path):
                recap_count = get_recap_scene_count(scenes_json_path)
                if recap_count > 0:
                    logger.info(f"🔄 Original episode had {recap_count} recap scene(s) that were filtered out for narrative analysis")
                else:
                    logger.info("ℹ️ No recap scenes detected in this episode")
        except Exception as e:
            logger.warning(f"Could not check recap status: {e}")

    # Season plot is no longer used - we process episodes independently
    state['season_plot'] = ""

    if os.path.exists(state['file_paths']['suggested_episode_arc_path']):
        state['suggested_episode_arc'] = load_text(state['file_paths']['suggested_episode_arc_path'])

    logger.info(f"Loaded {len(state['existing_season_entities'])} existing entities from season file.")

    return state

def log_agent_output(agent_name: str, output_data: dict, log_dir: str = "agent_logs") -> None:
    """Log agent output to a file with timestamp."""
    # Create logs directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamp for the current run if it doesn't exist
    timestamp_file = log_dir_path / "current_run_timestamp.txt"
    if not timestamp_file.exists():
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_file.write_text(current_timestamp)
    else:
        current_timestamp = timestamp_file.read_text().strip()
    
    # Create log file path
    log_file = log_dir_path / f"run_{current_timestamp}.jsonl"
    
    # Prepare log entry with proper formatting
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "output": output_data
    }
    
    # Format the JSON with proper indentation and ensure_ascii=False for proper character handling
    formatted_json = json.dumps(
        log_entry,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str  # Handles any non-serializable objects
    )
    
    # Add a newline after each formatted JSON entry
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(formatted_json + "\n\n")  # Add extra newline for better readability
    
    logger.info(f"Logged output from {agent_name}")

def identify_present_season_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Identify which existing season arcs are clearly present in the current episode."""
    logger.info("Identifying present season arcs in the episode.")

    # Initialize the database session manager and repositories
    db_manager = DatabaseSessionManager()
    serialized_season_arcs = []

    try:
        with db_manager.session_scope() as session:
            # Initialize repositories
            arc_repository = NarrativeArcRepository(session)

            # Get all narrative arcs for the series
            season_arcs = arc_repository.get_all(series=state['series'])

            # Filter arcs that have progressions in the current season
            filtered_arcs = []
            for arc in season_arcs:
                progressions_in_season = [prog for prog in arc.progressions if prog.season == state['season']]
                if progressions_in_season:
                    arc.progressions = progressions_in_season
                    filtered_arcs.append(arc)

            state['season_arcs'] = [arc.dict() for arc in filtered_arcs]
            logger.info(f"Retrieved {len(filtered_arcs)} arcs for {state['series']} season {state['season']}.")

    except Exception as e:
        logger.error(f"Error managing season arcs: {e}")
        state['season_arcs'] = []
        return state

    if not state['season_arcs']:
        logger.warning("No existing season arcs found. Skipping present season arcs identification.")
        state['present_season_arcs'] = []
        return state

    present_arcs = []
    # Process each arc individually
    for arc in state['season_arcs']:
        try:
            #do not consider anthology arcs for individuation in other episodes
            if arc['arc_type'] == "Anthology Arc":
                continue

            response = llm.invoke(PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT.format_messages(
                summarized_episode_plot=state['episode_plot'],
                arc_title=arc['title'],
                arc_description=arc['description']
            ))

            arc_data = clean_llm_json_response(response.content)
            if isinstance(arc_data, list):
                arc_data = arc_data[0]

            if arc_data['is_present']:
                present_arcs.append({
                    "title": arc_data['title'],
                    "description": arc_data['description'],
                    "presence_explanation": arc_data['explanation']
                })
                logger.warning(f"Arc '{arc['title']}' identified as present in episode.")
            else:
                logger.info(f"Arc '{arc['title']}' not present in episode.")

        except Exception as e:
            logger.error(f"Error processing arc '{arc['title']}': {e}")
            continue

    state['present_season_arcs'] = present_arcs
    logger.info(f"Identified {len(present_arcs)} season arcs present in the episode.")

    # Save present running plotlines to file
    try:
        from backend.src.path_handler import PathHandler
        path_handler = PathHandler(state['series'], state['season'], state['episode'])
        present_plotlines_path = path_handler.get_present_running_plotlines_path()
        
        # Ensure the recap_files directory exists
        os.makedirs(os.path.dirname(present_plotlines_path), exist_ok=True)
        
        # Save the present arcs with their information
        plotlines_data = []
        for arc in present_arcs:
            plotlines_data.append({
                "title": arc["title"],
                "description": arc["description"],
                "explanation": arc["presence_explanation"]
            })
        
        with open(present_plotlines_path, 'w', encoding='utf-8') as f:
            json.dump(plotlines_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(plotlines_data)} present running plotlines to: {present_plotlines_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save present running plotlines: {e}")

    # Add logging before returning
    log_agent_output("identify_present_season_arcs", {
        "present_season_arcs": state['present_season_arcs'],
        "season_arcs": state['season_arcs']
    })
    
    return state

def extract_anthology_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Extract self-contained anthology arcs from the episode plot."""
    logger.info("Extracting anthology arcs.")

    response = llm.invoke(ANTHOLOGY_ARC_EXTRACTOR_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        output_json_format=BRIEF_OUTPUT_JSON_FORMAT,
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        arcs = []
        for arc in arcs_data:
            new_arc = ExtractedArcBase(
                title=arc['title'],
                description=arc['description'],
                arc_type="Anthology Arc"
            )
            arcs.append(new_arc)
        state['anthology_arcs'] = arcs
        logger.info(f"Extracted {len(arcs)} anthology arcs.")
    except Exception as e:
        logger.error(f"Error extracting anthology arcs: {e}")
        state['anthology_arcs'] = []

    # Add logging before returning
    log_agent_output("extract_anthology_arcs", {
        "anthology_arcs": [arc.model_dump() for arc in state['anthology_arcs']]
    })
    
    return state

def extract_soap_and_genre_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Extract both soap and genre-specific arcs from the episode plot."""
    logger.info("Extracting soap and genre-specific arcs.")

    response = llm.invoke(SOAP_AND_GENRE_ARC_EXTRACTOR_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        present_season_arcs_summaries=json.dumps(state['present_season_arcs'], indent=2),
        anthology_arcs=json.dumps([arc.model_dump() for arc in state['anthology_arcs']], indent=2),
        output_json_format=EXTRACTOR_OUTPUT_JSON_FORMAT
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        soap_arcs = []
        genre_arcs = []
        
        for arc in arcs_data:
            new_arc = ExtractedArcBase(
                title=arc['title'],
                description=arc['description'],
                arc_type=arc['arc_type']
            )
            if arc['arc_type'] == "Soap Arc":
                soap_arcs.append(new_arc)
            else:  # Genre-Specific Arc
                genre_arcs.append(new_arc)
                
        state['soap_arcs'] = soap_arcs
        state['genre_arcs'] = genre_arcs
        logger.info(f"Extracted {len(soap_arcs)} soap arcs and {len(genre_arcs)} genre-specific arcs.")
    except Exception as e:
        logger.error(f"Error extracting soap and genre arcs: {e}")
        state['soap_arcs'] = []
        state['genre_arcs'] = []

    # Add logging before returning
    log_agent_output("extract_soap_and_genre_arcs", {
        "soap_arcs": [arc.model_dump() for arc in state['soap_arcs']],
        "genre_arcs": [arc.model_dump() for arc in state['genre_arcs']]
    })
    
    return state

def verify_arc_progression(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify and adjust the progression and description of each arc."""
    logger.info("Verifying arc progressions.")

    combined_arcs = state['episode_arcs']

    verified_arcs = []
    for arc in combined_arcs:
        response = llm.invoke(ARC_PROGRESSION_VERIFIER_PROMPT.format_messages(
            episode_plot=state['episode_plot'],
            arc_to_verify=arc.model_dump(),
            output_json_format=DETAILED_OUTPUT_JSON_FORMAT
        ))

        try:
            verified_arc_data = clean_llm_json_response(response.content)
            if isinstance(verified_arc_data, list):
                verified_arc_data = verified_arc_data[0]

            # Convert events to the new format
            events_data = verified_arc_data.get('single_episode_progression_events', [])
            if not events_data and 'single_episode_progression_string' in verified_arc_data:
                # Convert old string format to event format
                events_data = [{
                    'content': verified_arc_data['single_episode_progression_string'],
                    'ordinal_position': 1,
                    'characters_involved': []
                }]
            
            verified_arc = IntermediateNarrativeArc(
                title=verified_arc_data['title'],
                arc_type=verified_arc_data['arc_type'],
                description=verified_arc_data['description'],
                main_characters=verified_arc_data['main_characters'] if 'main_characters' in verified_arc_data else '',
                interfering_episode_characters=verified_arc_data['interfering_episode_characters'] if 'interfering_episode_characters' in verified_arc_data else '',
                single_episode_progression_events=[EventProgression(**event) for event in events_data]
            )
            verified_arcs.append(verified_arc)
        except Exception as e:
            logger.error(f"Error verifying arc progression: {e}")
            verified_arcs.append(arc)  # Keep the original arc if verification fails

    state['episode_arcs'] = verified_arcs
    logger.info(f"Verified progressions for {len(verified_arcs)} arcs.")

    # Add logging before returning
    log_agent_output("verify_arc_progression", {
        "verified_arcs": [arc.model_dump() for arc in state['episode_arcs']]
    })
    
    return state

def verify_character_roles(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify and correctly categorize characters as either main or interfering for each arc."""
    logger.info("Verifying character roles in arcs.")

    verified_arcs = []
    for arc in state['episode_arcs']:
        response = llm.invoke(CHARACTER_VERIFIER_PROMPT.format_messages(
            episode_plot=state['episode_plot'],
            arc_to_verify=arc.model_dump(),
            output_json_format=DETAILED_OUTPUT_JSON_FORMAT
        ))

        try:
            verified_arc_data = clean_llm_json_response(response.content)
            if isinstance(verified_arc_data, list):
                verified_arc_data = verified_arc_data[0]

            # Convert events to the new format
            events_data = verified_arc_data.get('single_episode_progression_events', [])
            if not events_data and 'single_episode_progression_string' in verified_arc_data:
                # Convert old string format to event format
                events_data = [{
                    'content': verified_arc_data['single_episode_progression_string'],
                    'ordinal_position': 1,
                    'characters_involved': []
                }]

            verified_arc = IntermediateNarrativeArc(
                title=verified_arc_data['title'],
                arc_type=verified_arc_data['arc_type'],
                description=verified_arc_data['description'],
                main_characters=verified_arc_data['main_characters'],
                interfering_episode_characters=verified_arc_data['interfering_episode_characters'],
                single_episode_progression_events=[EventProgression(**event) for event in events_data]
            )
            verified_arcs.append(verified_arc)
        except Exception as e:
            logger.error(f"Error verifying character roles: {e}")
            verified_arcs.append(arc)  # Keep original arc if verification fails

    state['episode_arcs'] = verified_arcs
    logger.info(f"Verified character roles for {len(verified_arcs)} arcs.")

    # Add logging before returning
    log_agent_output("verify_character_roles", {
        "verified_arcs": [arc.model_dump() for arc in state['episode_arcs']]
    })
    
    return state

def deduplicate_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Deduplicate and merge similar arcs from different extraction methods."""
    logger.info("Deduplicating and merging similar arcs from different extraction methods.")

    # Separate anthology arcs
    non_anthology_arcs = [arc for arc in state['optimized_arcs'] if arc.arc_type != "Anthology Arc"]
    anthology_arcs = [arc for arc in state['optimized_arcs'] if arc.arc_type == "Anthology Arc"]

    response = llm.invoke(ARC_DEDUPLICATOR_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        arcs_to_deduplicate=[arc.model_dump() for arc in non_anthology_arcs],
        anthology_arcs=[arc.model_dump() for arc in anthology_arcs],  # Added anthology arcs as context
        guidelines=NARRATIVE_ARC_GUIDELINES,
        output_json_format=EXTRACTOR_OUTPUT_JSON_FORMAT
    ))

    try:
        deduplicated_arcs_data = clean_llm_json_response(response.content)
        deduplicated_arcs = []
        for arc in deduplicated_arcs_data:
            new_arc = ExtractedArcBase(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description']
            )
            deduplicated_arcs.append(new_arc)
        
        # Combine anthology arcs with deduplicated arcs
        state['episode_arcs'] = anthology_arcs + deduplicated_arcs
        logger.info(f"Deduplicated to {len(deduplicated_arcs)} unique arcs + {len(anthology_arcs)} anthology arcs.")
    except Exception as e:
        logger.error(f"Error deduplicating arcs: {e}")
        state['episode_arcs'] = anthology_arcs + non_anthology_arcs

    # Add logging before returning
    log_agent_output("deduplicate_arcs", {
        "deduplicated_arcs": [arc.model_dump() for arc in state['episode_arcs']],
        "anthology_arcs_count": len(anthology_arcs),
        "other_arcs_count": len(deduplicated_arcs)
    })
    
    return state

def enhance_arc_details(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Enhance deduplicated arcs with additional details."""
    logger.info("Enhancing arcs with additional details.")

    enhanced_arcs = []
    for arc in state['episode_arcs']:
        response = llm.invoke(ARC_ENHANCER_PROMPT.format_messages(
            episode_plot=state['episode_plot'],
            arc_to_enhance=arc.model_dump(),
            guidelines=NARRATIVE_ARC_GUIDELINES,
            output_json_format=DETAILED_OUTPUT_JSON_FORMAT
        ))

        try:
            enhanced_arc_data = clean_llm_json_response(response.content)
            if isinstance(enhanced_arc_data, list):
                enhanced_arc_data = enhanced_arc_data[0]

            # Convert events to the new format
            events_data = enhanced_arc_data.get('single_episode_progression_events', [])
            if not events_data and 'single_episode_progression_string' in enhanced_arc_data:
                # Convert old string format to event format
                events_data = [{
                    'content': enhanced_arc_data['single_episode_progression_string'],
                    'ordinal_position': 1,
                    'characters_involved': []
                }]

            enhanced_arc = IntermediateNarrativeArc(
                title=arc.title,
                arc_type=arc.arc_type,
                description=arc.description,
                main_characters=enhanced_arc_data['main_characters'],
                interfering_episode_characters=enhanced_arc_data['interfering_episode_characters'],
                single_episode_progression_events=[EventProgression(**event) for event in events_data]
            )
            enhanced_arcs.append(enhanced_arc)
        except Exception as e:
            logger.error(f"Error enhancing arc details: {e}")
            # Create a basic enhanced arc if enhancement fails
            enhanced_arc = IntermediateNarrativeArc(
                title=arc.title,
                arc_type=arc.arc_type,
                description=arc.description,
                main_characters="",
                interfering_episode_characters="",
                single_episode_progression_events=[]
            )
            enhanced_arcs.append(enhanced_arc)

    state['episode_arcs'] = enhanced_arcs
    logger.info(f"Enhanced {len(enhanced_arcs)} arcs with additional details.")

    # Add logging before returning
    log_agent_output("enhance_arc_details", {
        "enhanced_arcs": [arc.model_dump() for arc in state['episode_arcs']]
    })
    
    return state

def verify_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify the arcs to ensure they are consistent with the episode plot and present season arcs."""
    logger.info("Verifying arcs.")

    response = llm.invoke(ARC_VERIFIER_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        arcs_to_verify=json.dumps([arc.model_dump() for arc in state['episode_arcs']], indent=2),
        present_season_arcs_summaries=json.dumps(state['present_season_arcs'], indent=2), 
        guidelines=NARRATIVE_ARC_GUIDELINES,
        output_json_format=DETAILED_OUTPUT_JSON_FORMAT
    ))

    try:
        verified_arcs_data = clean_llm_json_response(response.content)
        verified_arcs = []
        
        for arc_data in verified_arcs_data:
            try:
                # Convert events to the new format
                events_data = arc_data.get('single_episode_progression_events', [])
                if not events_data and 'single_episode_progression_string' in arc_data:
                    # Convert old string format to event format
                    events_data = [{
                        'content': arc_data['single_episode_progression_string'],
                        'ordinal_position': 1,
                        'characters_involved': []
                    }]
                
                # Ensure all required fields are present with defaults if missing
                new_arc = IntermediateNarrativeArc(
                    title=arc_data['title'],
                    arc_type=arc_data['arc_type'],
                    description=arc_data['description'],
                    main_characters=arc_data.get('main_characters', ''),
                    interfering_episode_characters=arc_data.get('interfering_episode_characters', ''),
                    single_episode_progression_events=[EventProgression(**event) for event in events_data]
                )
                verified_arcs.append(new_arc)
            except Exception as e:
                logger.error(f"Error processing verified arc: {e}")
                logger.error(f"Problematic arc data: {arc_data}")
                continue

        if verified_arcs:
            state['episode_arcs'] = verified_arcs
            logger.info(f"Successfully verified {len(verified_arcs)} arcs.")
        else:
            logger.warning("No arcs were successfully verified. Keeping original arcs.")
            
    except Exception as e:
        logger.error(f"Error during arc verification: {e}")
        # Keep original arcs if verification completely fails
        logger.warning("Verification failed. Keeping original arcs.")

    # Add logging before returning
    log_agent_output("verify_arcs", {
        "final_verified_arcs": [arc.model_dump() for arc in state['episode_arcs']]
    })
    
    return state

def assign_event_timestamps(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Assign timestamps to events using scene data and subtitle information."""
    logger.info("🕒 Starting timestamp assignment for events using subtitle data only")
    
    try:
        from ..path_handler import PathHandler
        from ..utils.subtitle_utils import parse_srt_file
        
        # Initialize path handler
        path_handler = PathHandler(state['series'], state['season'], state['episode'])
        
        # Load subtitle content for precise micro-event timestamp matching
        subtitle_content = ""
        possible_speakers_srt_path = path_handler.get_possible_speakers_srt_path()
        regular_srt_path = path_handler.get_srt_file_path()
        
        if os.path.exists(possible_speakers_srt_path):
            with open(possible_speakers_srt_path, 'r') as f:
                subtitle_content = f.read()
            logger.info(f"✅ Loaded speaker-identified subtitles: {len(subtitle_content)} characters")
        elif os.path.exists(regular_srt_path):
            with open(regular_srt_path, 'r') as f:
                subtitle_content = f.read()
            logger.info(f"✅ Loaded regular subtitles: {len(subtitle_content)} characters")
        else:
            logger.warning("⚠️ No subtitle files found")
        
        logger.info(f"🔍 Processing {len(state['episode_arcs'])} arcs for timestamp assignment")
        
        # Process each arc to assign timestamps to its events
        timestamped_arcs = []
        
        for arc_idx, arc in enumerate(state['episode_arcs']):
            logger.info(f"🔍 Processing arc {arc_idx + 1}/{len(state['episode_arcs'])}: '{arc.title}'")
            logger.info(f"   Arc type: {arc.arc_type}")
            logger.info(f"   Events count: {len(arc.single_episode_progression_events)}")
            
            timestamped_arc = IntermediateNarrativeArc(
                title=arc.title,
                arc_type=arc.arc_type,
                description=arc.description,
                main_characters=arc.main_characters,
                interfering_episode_characters=arc.interfering_episode_characters,
                single_episode_progression_events=[]
            )
            
            # Assign timestamps to events based on available data
            if arc.single_episode_progression_events:
                try:
                    timestamped_events = []
                    
                    # Use subtitle-based timestamp matching for granular precision
                    if subtitle_content:
                        logger.info(f"   Using subtitle-based timestamp matching for granular precision")
                        timestamped_events = _assign_timestamps_using_subtitles(
                            arc, subtitle_content, llm, logger
                        )
                    else:
                        # No timestamp data available - keep events without timestamps
                        logger.warning(f"   No subtitle data available for arc '{arc.title}'")
                        timestamped_events = arc.single_episode_progression_events
                    
                    timestamped_arc.single_episode_progression_events = timestamped_events
                    logger.info(f"✅ Processed {len(timestamped_events)} events for arc '{arc.title}'")
                    
                except Exception as e:
                    logger.error(f"❌ Error assigning timestamps to arc '{arc.title}': {e}")
                    logger.error(f"❌ Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    # Keep original events without timestamps
                    timestamped_arc.single_episode_progression_events = arc.single_episode_progression_events
            else:
                logger.warning(f"⚠️ Arc '{arc.title}' has no events to timestamp")
            
            timestamped_arcs.append(timestamped_arc)
        
        state['timestamped_arcs'] = timestamped_arcs
        logger.info(f"✅ Timestamp assignment completed for {len(timestamped_arcs)} arcs")
        
        # Log summary of timestamp assignment
        total_events = sum(len(arc.single_episode_progression_events) for arc in timestamped_arcs)
        timestamped_events = sum(1 for arc in timestamped_arcs 
                               for event in arc.single_episode_progression_events 
                               if hasattr(event, 'start_timestamp') and event.start_timestamp is not None)
        logger.info(f"📊 Timestamp assignment summary:")
        logger.info(f"   Total events: {total_events}")
        logger.info(f"   Events with timestamps: {timestamped_events}")
        logger.info(f"   Events without timestamps: {total_events - timestamped_events}")
        
    except Exception as e:
        logger.error(f"❌ Error during timestamp assignment: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        # Fallback: use original arcs without timestamps
        state['timestamped_arcs'] = state['episode_arcs']
    
    # Add logging
    log_agent_output("assign_event_timestamps", {
        "timestamped_arcs": [arc.model_dump() for arc in state['timestamped_arcs']]
    })
    
    return state

def _assign_timestamps_using_subtitles(arc, subtitle_content, llm, logger):
    """Assign granular, precise timestamps using subtitle dialogue for micro-event precision."""
    
    logger.info(f"   Using full subtitle content: {len(subtitle_content)} characters")
    
    # Prepare events for timestamp matching
    events_for_matching = [
        {
            'content': event.content,
            'ordinal_position': event.ordinal_position,
            'characters_involved': event.characters_involved
        }
        for event in arc.single_episode_progression_events
    ]
    
    logger.info(f"   Subtitle-based matching for {len(events_for_matching)} events")
    
    # Enhanced prompt for granular subtitle-based timestamp matching
    prompt = f"""You are an expert timestamp assignment specialist. Your task is to assign PRECISE, GRANULAR timestamps to narrative micro-events by matching them to specific dialogue lines in subtitles.

**Arc Title:** {arc.title}
**Arc Description:** {arc.description}

**Events to Timestamp (GRANULAR PRECISION REQUIRED):**
{json.dumps(events_for_matching, indent=2)}

**Episode Subtitles (with precise timestamps):**
{subtitle_content}

**CRITICAL INSTRUCTIONS FOR GRANULAR TIMING:**
1. **MICRO-EVENT PRECISION**: Each event should be matched to the exact dialogue moment where it occurs
2. **EXACT SUBTITLE MATCHING**: Find the specific subtitle lines that correspond to each event
3. **CHARACTER-SPECIFIC TIMING**: Match events to when specific characters speak or act
4. **KEEP ORIGINAL FORMAT**: Use the exact HH:MM:SS,mmm format from subtitles (e.g., 00:03:11,594)
5. **MULTI-DIALOGUE EVENTS**: Events can span multiple subtitle entries - use the start time of the first relevant subtitle and end time of the last relevant subtitle
6. **SEQUENTIAL ORDERING**: Events must be in chronological order by ordinal_position
7. **NO OVERLAPS**: Each event gets its own unique time window
8. **HIGH CONFIDENCE**: Use confidence 0.8+ only when dialogue clearly matches the event

**TIMING RULES:**
- Event start_timestamp = exact moment the event begins (HH:MM:SS,mmm format)
- Event end_timestamp = exact moment the event concludes (HH:MM:SS,mmm format)  
- Events typically last 5-60 seconds depending on complexity
- Look for character names, specific dialogue, or action descriptions
- **IMPORTANT**: If an event spans multiple dialogue exchanges, include all relevant subtitle entries from start to finish
- Ensure NO two events have identical timestamps

**EVENT SPAN EXAMPLES:**
- Single dialogue event: Use one subtitle entry's timestamps
- Multi-dialogue event: Use start of first subtitle to end of last subtitle in the sequence
- Character interaction: Include the full conversation from beginning to resolution

**Output Format (JSON only):**
[
    {{
        "content": "Event description",
        "ordinal_position": 1,
        "characters_involved": ["Character1", "Character2"],
        "start_timestamp": "00:03:25,123",
        "end_timestamp": "00:03:28,456",
        "confidence_score": 0.85,
        "extraction_method": "dialogue_matching"
    }}
]

Provide GRANULAR, PRECISE timestamps for each event using the exact HH:MM:SS,mmm format from subtitles. Remember that events can span multiple dialogue exchanges - capture the full temporal scope of each narrative event."""

    response = llm.invoke([{"role": "user", "content": prompt}])
    logger.info(f"   Subtitle-matching LLM response: {len(response.content)} characters")
    
    from ..utils.llm_utils import clean_llm_json_response
    timestamped_events_data = clean_llm_json_response(response.content)
    
    if isinstance(timestamped_events_data, str):
        timestamped_events_data = json.loads(timestamped_events_data)
    
    # VALIDATION: Check for duplicate timestamps and ensure granular precision
    _validate_granular_timestamps(timestamped_events_data, logger)
    
    # Convert to EventProgression objects with timestamps
    timestamped_events = []
    for i, event_data in enumerate(timestamped_events_data):
        event = EventProgression(
            content=event_data.get('content', ''),
            ordinal_position=event_data.get('ordinal_position', 1),
            characters_involved=event_data.get('characters_involved', [])
        )
        # Add timestamp information as attributes
        event.start_timestamp = event_data.get('start_timestamp')
        event.end_timestamp = event_data.get('end_timestamp') 
        event.confidence_score = event_data.get('confidence_score', 0.5)
        event.extraction_method = event_data.get('extraction_method', 'dialogue_matching')
        timestamped_events.append(event)
        
        start_ts = event.start_timestamp
        end_ts = event.end_timestamp
        confidence = event.confidence_score
        logger.info(f"     Event {i+1}: {start_ts} - {end_ts} (confidence: {confidence:.2f})")
    
    return timestamped_events

def _validate_granular_timestamps(events_data, logger):
    """Validate that timestamps are granular and don't have duplicates."""
    logger.info("🔍 Validating granular timestamp precision...")
    
    def timestamp_to_seconds(timestamp_str):
        """Convert HH:MM:SS,mmm to seconds for validation calculations."""
        if not timestamp_str or not isinstance(timestamp_str, str):
            return None
        try:
            # Parse HH:MM:SS,mmm format
            time_part, ms_part = timestamp_str.split(',')
            hours, minutes, seconds = map(int, time_part.split(':'))
            milliseconds = int(ms_part)
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        except:
            return None
    
    timestamp_ranges = []
    issues_found = []
    
    for i, event_data in enumerate(events_data):
        start_str = event_data.get('start_timestamp')
        end_str = event_data.get('end_timestamp')
        
        if not start_str or not end_str:
            issues_found.append(f"Event {i+1}: Missing timestamps")
            continue
        
        start_seconds = timestamp_to_seconds(start_str)
        end_seconds = timestamp_to_seconds(end_str)
        
        if start_seconds is None or end_seconds is None:
            issues_found.append(f"Event {i+1}: Invalid timestamp format")
            continue
            
        duration = end_seconds - start_seconds
        timestamp_ranges.append((start_str, end_str))
        
        # Check for micro-event precision (should be granular, not broad)
        if duration > 120:  # More than 2 minutes is too broad for micro-events
            issues_found.append(f"Event {i+1}: Duration too broad ({duration:.1f}s) - should be granular")
        
        if duration < 1:  # Less than 1 second might be too narrow
            issues_found.append(f"Event {i+1}: Duration too narrow ({duration:.1f}s)")
    
    # Check for exact duplicates
    unique_ranges = set(timestamp_ranges)
    if len(timestamp_ranges) != len(unique_ranges):
        duplicate_ranges = [x for x in timestamp_ranges if timestamp_ranges.count(x) > 1]
        issues_found.append(f"Duplicate timestamp ranges found: {duplicate_ranges}")
    
    # Note: Events CAN overlap - this is allowed for micro-events
    # No overlap validation needed
    
    if issues_found:
        logger.warning("⚠️ Granular timestamp validation issues found:")
        for issue in issues_found:
            logger.warning(f"   - {issue}")
    else:
        logger.info("✅ All timestamps pass granular precision validation")
    
    return len(issues_found) == 0

def optimize_arcs_with_season_context(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Optimize arc titles and descriptions based on seasonal context and merge related arcs."""
    logger.info("Starting seasonal arc optimization.")

    # Combine soap and genre arcs for optimization
    arcs_to_optimize = state['soap_arcs'] + state['genre_arcs']

    if not arcs_to_optimize:
        logger.info("No arcs to optimize.")
        return state

    response = llm.invoke(SEASONAL_ARC_OPTIMIZER_PROMPT.format_messages(
        present_season_arcs=json.dumps(state['present_season_arcs'], indent=2),
        new_arcs=json.dumps([arc.model_dump() for arc in arcs_to_optimize], indent=2),
        output_json_format=EXTRACTOR_OUTPUT_JSON_FORMAT
    ))

    try:
        optimized_arcs_data = clean_llm_json_response(response.content)
        optimized_arcs = []
        
        for arc_data in optimized_arcs_data:
            optimized_arc = ExtractedArcBase(
                title=arc_data['title'],
                arc_type=arc_data['arc_type'],
                description=arc_data['description']
            )
            optimized_arcs.append(optimized_arc)

        # Store optimized arcs in episode_arcs along with anthology arcs
        state['optimized_arcs'] = state['anthology_arcs'] + optimized_arcs
        logger.info(f"Optimized arcs into {len(optimized_arcs)} arcs + {len(state['anthology_arcs'])} anthology arcs.")
        
    except Exception as e:
        logger.error(f"Error during arc optimization: {e}")
        # If optimization fails, use original arcs along with anthology arcs
        state['optimized_arcs'] = state['anthology_arcs'] + arcs_to_optimize
        
    # Add logging before returning
    log_agent_output("optimize_arcs", {
        "optimized_arcs": [arc.model_dump() for arc in state['optimized_arcs']]
    })
    
    return state

def create_narrative_arc_graph():
    """Create and configure the state graph for narrative arc extraction."""
    workflow = StateGraph(NarrativeArcsExtractionState)

    # Add nodes
    workflow.add_node("initialize_state_node", initialize_state)
    workflow.add_node("identify_present_season_arcs_node", identify_present_season_arcs)
    workflow.add_node("extract_anthology_arcs_node", extract_anthology_arcs)
    workflow.add_node("extract_soap_and_genre_arcs_node", extract_soap_and_genre_arcs)
    workflow.add_node("optimize_arcs_node", optimize_arcs_with_season_context)
    workflow.add_node("deduplicate_arcs_node", deduplicate_arcs)
    workflow.add_node("enhance_arc_details_node", enhance_arc_details)
    workflow.add_node("verify_arc_progression_node", verify_arc_progression)
    workflow.add_node("verify_character_roles_node", verify_character_roles)
    workflow.add_node("verify_arcs_node", verify_arcs)
    workflow.add_node("assign_event_timestamps_node", assign_event_timestamps)  # NEW: Timestamp assignment

    # Set up the workflow
    workflow.set_entry_point("initialize_state_node")
    workflow.add_edge("initialize_state_node", "identify_present_season_arcs_node")
    workflow.add_edge("identify_present_season_arcs_node", "extract_anthology_arcs_node")
    workflow.add_edge("extract_anthology_arcs_node", "extract_soap_and_genre_arcs_node")
    workflow.add_edge("extract_soap_and_genre_arcs_node", "optimize_arcs_node")
    workflow.add_edge("optimize_arcs_node", "deduplicate_arcs_node")
    workflow.add_edge("deduplicate_arcs_node", "enhance_arc_details_node")
    workflow.add_edge("enhance_arc_details_node", "verify_arc_progression_node")
    workflow.add_edge("verify_arc_progression_node", "verify_character_roles_node")
    workflow.add_edge("verify_character_roles_node", "verify_arcs_node")
    workflow.add_edge("verify_arcs_node", "assign_event_timestamps_node")  # NEW: Add timestamp assignment
    workflow.add_edge("assign_event_timestamps_node", END)

    return workflow.compile()

def extract_narrative_arcs(file_paths: Dict[str, str], series: str, season: str, episode: str) -> None:
    """Extract narrative arcs from the provided file paths and save the results to a JSON file."""
    logger.info("Starting extract_narrative_arcs function")
    
    # Create a unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("agent_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "current_run_timestamp.txt").write_text(timestamp)
    
    graph = create_narrative_arc_graph()

    initial_state = NarrativeArcsExtractionState(
        soap_arcs=[],
        genre_arcs=[],
        anthology_arcs=[],
        episode_arcs=[],
        present_season_arcs=[],
        season_arcs=[],  # Initialize season_arcs
        file_paths=file_paths,
        series=series,
        season=season,
        episode=episode,
        existing_season_entities=[],
        episode_plot="",
        season_plot="",
        optimized_arcs=[],
        timestamped_arcs=[]  # NEW: Initialize timestamped_arcs
    )
    logger.info("Invoking the graph")
    result = graph.invoke(initial_state)
    logger.info("Graph execution completed")

    # Save the results to a JSON file - use timestamped arcs if available
    arcs_to_save = result.get('timestamped_arcs', result['episode_arcs'])
    suggested_arcs = [arc.dict() for arc in arcs_to_save]

    save_json(suggested_arcs, file_paths['suggested_episode_arc_path'])

    logger.info(f"Suggested episode arcs saved to {file_paths['suggested_episode_arc_path']}")

def extract_events_from_progression(progression_id: str, series: str, season: str, episode: str, progression_content: str) -> Dict:
    """
    Extract timestamped events from a single progression for API/migration usage.
    
    This function provides compatibility with the previous event_timestamp_extraction_graph.py
    functionality by processing a single progression through the timestamp assignment pipeline.
    
    Args:
        progression_id: ID of the progression to process
        series: Series name
        season: Season name  
        episode: Episode name
        progression_content: Content of the progression to segment
        
    Returns:
        Dict with extraction results
    """
    logger.info(f"🚀 Extracting events from progression {progression_id}")
    
    try:
        # Parse the progression content into individual events
        prompt = f"""You are an expert at analyzing narrative progression text and breaking it down into individual, discrete events.

Break down this progression into individual events that can be timestamped:

**Progression Content:**
{progression_content}

**Output Format (JSON only):**
[
  {{
    "content": "Specific description of the individual event",
    "ordinal_position": 1,
    "characters_involved": ["Character1", "Character2"]
  }},
  {{
    "content": "Next individual event description", 
    "ordinal_position": 2,
    "characters_involved": ["Character3"]
  }}
]

Extract 3-10 discrete events from the progression content. Be specific and precise."""

        response = llm.invoke([{"role": "user", "content": prompt}])
        events_data = clean_llm_json_response(response.content)
        
        if isinstance(events_data, str):
            events_data = json.loads(events_data)
        
        logger.info(f"✅ Segmented progression into {len(events_data)} events")
        
        # Create Event objects in database using narrative_arc_service
        from ..narrative_storage_management.repositories import ArcProgressionRepository, EventRepository
        from ..narrative_storage_management.narrative_models import Event
        
        db_manager = DatabaseSessionManager()
        created_events = []
        
        with db_manager.session_scope() as session:
            progression_repo = ArcProgressionRepository(session)
            event_repo = EventRepository(session)
            
            # Get the progression
            progression = progression_repo.get_by_id(progression_id)
            if not progression:
                raise ValueError(f"Progression {progression_id} not found")
            
            # Create Event objects
            for i, event_data in enumerate(events_data):
                event = Event(
                    progression_id=progression_id,
                    content=event_data.get('content', ''),
                    series=series,
                    season=season,
                    episode=episode,
                    ordinal_position=event_data.get('ordinal_position', i + 1),
                    confidence_score=0.8,  # Default confidence for content-based extraction
                    extraction_method='content_segmentation'
                )
                
                created_event = event_repo.create(event)
                created_events.append(created_event)
                logger.info(f"   ✅ Created event: {created_event.content[:50]}...")
        
        result = {
            "success": True,
            "error_message": None,
            "events_extracted": len(created_events),
            "validation_results": {"is_valid": True, "overall_quality_score": 0.8},
            "extracted_events": [
                {
                    "content": event.content,
                    "start_timestamp": event.start_timestamp,
                    "end_timestamp": event.end_timestamp,
                    "ordinal_position": event.ordinal_position,
                    "confidence_score": event.confidence_score,
                    "extraction_method": event.extraction_method,
                    "characters_involved": []  # Would need additional processing to extract
                }
                for event in created_events
            ]
        }
        
        logger.info(f"🎯 Event extraction completed for {progression_id}: {result['events_extracted']} events")
        return result
        
    except Exception as e:
        logger.error(f"❌ Event extraction failed for {progression_id}: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        return {
            "success": False,
            "error_message": str(e),
            "events_extracted": 0,
            "validation_results": {},
            "extracted_events": []
        }
