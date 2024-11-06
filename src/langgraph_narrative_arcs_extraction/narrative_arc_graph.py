# narrative_arc_graph.py

from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
import os
import json

from regex import D

from src.utils.llm_utils import clean_llm_json_response
from src.utils.logger_utils import setup_logging
from src.utils.text_utils import load_text, save_json
from src.ai_models.ai_models import get_llm, LLMType
from src.plot_processing.plot_ner_entity_extraction import EntityLink, normalize_names
# Import the new repositories and models
from src.narrative_storage.repositories import (
    DatabaseSessionManager,
    NarrativeArcRepository,
)
from src.narrative_storage.narrative_models import NarrativeArc, ArcProgression
import logging

logger = setup_logging(__name__)
llm = get_llm(LLMType.INTELLIGENT)
cheap_llm = get_llm(LLMType.CHEAP)
from pydantic import BaseModel, Field

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

class IntermediateNarrativeArc(BaseModel):
    """Model representing an intermediate narrative arc during extraction process."""
    title: str = Field(..., description="The title of the narrative arc")
    arc_type: str = Field(..., description="Type of the arc such as 'Soap Arc'/'Genre-Specific Arc'/'Anthology Arc'")
    description: str = Field(..., description="A brief description of the narrative arc")
    main_characters: str = Field(..., description="Main characters involved in this arc")
    interfering_episode_characters: str = Field(..., description="Interfering characters involved in this arc")
    single_episode_progression_string: str = Field("", description="The progression of the arc for the current episode")

    class Config:
        populate_by_name = True

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
    summarized_plot: str
    season_plot: str
    optimized_arcs: List[IntermediateNarrativeArc]

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
        state['episode_plot'] = load_text(state['file_paths']['episode_plot_path'])

    if os.path.exists(state['file_paths']['summarized_plot_path']):
        state['summarized_plot'] = load_text(state['file_paths']['summarized_plot_path'])

    if os.path.exists(state['file_paths']['season_plot_path']):
        state['season_plot'] = load_text(state['file_paths']['season_plot_path'])

    if os.path.exists(state['file_paths']['suggested_episode_arc_path']):
        state['suggested_episode_arc'] = load_text(state['file_paths']['suggested_episode_arc_path'])

    logger.info(f"Loaded {len(state['existing_season_entities'])} existing entities from season file.")

    return state

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
                # Assuming progressions are already loaded due to selectinload in the repository
                progressions_in_season = [prog for prog in arc.progressions if prog.season == state['season']]
                if progressions_in_season:
                    # Update arc's progressions to only include those in the current season
                    arc.progressions = progressions_in_season
                    filtered_arcs.append(arc)

            state['existing_season_arcs'] = [arc.dict() for arc in filtered_arcs]

            for arc in filtered_arcs:
                # Serialize the arc and its progressions
                serialized_arc = arc.dict()
                serialized_arc['progressions'] = [prog.dict() for prog in arc.progressions]
                serialized_season_arcs.append(serialized_arc)

            state['season_arcs'] = serialized_season_arcs
            logger.info(f"Retrieved {len(serialized_season_arcs)} arcs for {state['series']} season {state['season']}.")

    except Exception as e:
        logger.error(f"Error managing season arcs: {e}")
        state['season_arcs'] = []

    # Prepare existing season arcs summaries
    existing_season_arcs_summaries = [
        {"title": arc['title'], "description": arc['description']}
        for arc in serialized_season_arcs
    ]

    if len(existing_season_arcs_summaries) == 0:
        logger.warning("No existing season arcs found. Skipping present season arcs identification.")
        state['present_season_arcs'] = []
        return state

    response = llm.invoke(PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT.format_messages(
        summarized_episode_plot=state['summarized_plot'],
        existing_season_arcs_summaries=json.dumps(existing_season_arcs_summaries, indent=2),
        output_json_format=PRESENT_SEASON_ARCS_OUTPUT_JSON_FORMAT
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        present_arcs = []
        for arc in arcs_data:
            present_arcs.append({
                "title": arc['title'],
                "description": arc['description'],
                "presence_explanation": arc.get('presence_explanation', '')
            })
        state['present_season_arcs'] = present_arcs
        logger.info(f"Identified {len(present_arcs)} season arcs present in the episode.")
    except Exception as e:
        logger.error(f"Error identifying present season arcs: {e}")
        state['present_season_arcs'] = []

    return state

def extract_anthology_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Extract self-contained anthology arcs from the episode plot."""
    logger.info("Extracting anthology arcs.")

    response = llm.invoke(ANTHOLOGY_ARC_EXTRACTOR_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        season_plot=state['season_plot'],
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

            verified_arc = IntermediateNarrativeArc(
                title=verified_arc_data['title'],
                arc_type=verified_arc_data['arc_type'],
                description=verified_arc_data['description'],
                main_characters=verified_arc_data['main_characters'],
                interfering_episode_characters=verified_arc_data['interfering_episode_characters'],
                single_episode_progression_string=verified_arc_data['single_episode_progression_string']
            )
            verified_arcs.append(verified_arc)
        except Exception as e:
            logger.error(f"Error verifying arc progression: {e}")
            verified_arcs.append(arc)  # Keep the original arc if verification fails

    state['episode_arcs'] = verified_arcs
    logger.info(f"Verified progressions for {len(verified_arcs)} arcs.")
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

            verified_arc = IntermediateNarrativeArc(
                title=verified_arc_data['title'],
                arc_type=verified_arc_data['arc_type'],
                description=verified_arc_data['description'],
                main_characters=verified_arc_data['main_characters'],
                interfering_episode_characters=verified_arc_data['interfering_episode_characters'],
                single_episode_progression_string=verified_arc_data['single_episode_progression_string']
            )
            verified_arcs.append(verified_arc)
        except Exception as e:
            logger.error(f"Error verifying character roles: {e}")
            verified_arcs.append(arc)  # Keep original arc if verification fails

    state['episode_arcs'] = verified_arcs
    logger.info(f"Verified character roles for {len(verified_arcs)} arcs.")
    return state

def deduplicate_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Deduplicate and merge similar arcs from different extraction methods."""
    logger.info("Deduplicating and merging similar arcs from different extraction methods.")

    # Use optimized_arcs instead of combining all arcs
    all_arcs = state['optimized_arcs']

    response = llm.invoke(ARC_DEDUPLICATOR_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        arcs_to_deduplicate=[arc.model_dump() for arc in all_arcs],
        present_season_arcs_summaries=json.dumps(state['present_season_arcs'], indent=2),
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
        state['episode_arcs'] = deduplicated_arcs
        logger.info(f"Deduplicated to {len(deduplicated_arcs)} unique arcs from {len(all_arcs)} original arcs.")
    except Exception as e:
        logger.error(f"Error deduplicating arcs: {e}")
        state['episode_arcs'] = all_arcs  # Keep original arcs if deduplication fails

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

            enhanced_arc = IntermediateNarrativeArc(
                title=arc.title,
                arc_type=arc.arc_type,
                description=arc.description,
                main_characters=enhanced_arc_data['main_characters'],
                interfering_episode_characters=enhanced_arc_data['interfering_episode_characters'],
                single_episode_progression_string=enhanced_arc_data['single_episode_progression_string']
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
                single_episode_progression_string=""
            )
            enhanced_arcs.append(enhanced_arc)

    state['episode_arcs'] = enhanced_arcs
    logger.info(f"Enhanced {len(enhanced_arcs)} arcs with additional details.")
    return state

def verify_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify the arcs to ensure they are consistent with the episode plot and present season arcs."""
    logger.info("Verifying arcs.")

    response = llm.invoke(ARC_VERIFIER_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        season_plot=state['season_plot'],
        arcs_to_verify=json.dumps([arc.model_dump() for arc in state['episode_arcs']], indent=2),
        present_season_arcs_summaries=json.dumps(state['present_season_arcs'], indent=2),
        guidelines=NARRATIVE_ARC_GUIDELINES,
        output_json_format=DETAILED_OUTPUT_JSON_FORMAT
    ))

    try:
        verified_arcs_data = clean_llm_json_response(response.content)
        verified_arcs = []
        for arc in verified_arcs_data:
            new_arc = IntermediateNarrativeArc(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description'],
                main_characters=arc['main_characters'],
                interfering_episode_characters=arc['interfering_episode_characters'],
                single_episode_progression_string=arc['single_episode_progression_string']
            )
            verified_arcs.append(new_arc)

        logger.info(f"Verified {len(state['episode_arcs'])} arcs.")

        state['episode_arcs'] = verified_arcs

    except Exception as e:
        logger.error(f"Error verifying arcs: {e}")

    return state

def optimize_arcs_with_season_context(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Optimize arc titles and descriptions based on seasonal context and merge related arcs."""
    logger.info("Starting seasonal arc optimization.")

    # Combine soap and genre arcs for optimization
    arcs_to_optimize = state['soap_arcs'] + state['genre_arcs']

    if not arcs_to_optimize:
        logger.info("No arcs to optimize.")
        return state

    response = llm.invoke(SEASONAL_ARC_OPTIMIZER_PROMPT.format_messages(
        season_plot=state['season_plot'],
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
    workflow.add_edge("verify_arcs_node", END)

    return workflow.compile()

def extract_narrative_arcs(file_paths: Dict[str, str], series: str, season: str, episode: str) -> None:
    """Extract narrative arcs from the provided file paths and save the results to a JSON file."""
    logger.info("Starting extract_narrative_arcs function")
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
        summarized_plot="",
        season_plot="",
        optimized_arcs=[]
    )
    logger.info("Invoking the graph")
    result = graph.invoke(initial_state)
    logger.info("Graph execution completed")

    # Save the results to a JSON file
    suggested_arcs = [arc.dict() for arc in result['episode_arcs']]

    save_json(suggested_arcs, file_paths['suggested_episode_arc_path'])

    logger.info(f"Suggested episode arcs saved to {file_paths['suggested_episode_arc_path']}")
