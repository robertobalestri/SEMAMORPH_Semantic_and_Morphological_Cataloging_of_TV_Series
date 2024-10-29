# narrative_arc_graph.py

from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
import os
import json

from src.utils.llm_utils import clean_llm_json_response
from src.utils.logger_utils import setup_logging
from src.utils.text_utils import load_text, save_json
from src.ai_models.ai_models import get_llm, LLMType
from src.plot_processing.plot_ner_entity_extraction import EntityLink, normalize_names
from src.storage.narrative_arc_manager import NarrativeArcManager

logger = setup_logging(__name__)

from .prompts import (
    SEASONAL_NARRATIVE_ANALYZER_PROMPT,
    EPISODE_NARRATIVE_ANALYZER_PROMPT,
    PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT,
    ARC_EXTRACTOR_FROM_ANALYSIS_PROMPT,
    ARC_EXTRACTOR_FROM_PLOT_PROMPT,
    ARC_VERIFIER_PROMPT,
    NARRATIVE_ARC_GUIDELINES,
    ARC_PROGRESSION_VERIFIER_PROMPT,
    OUTPUT_JSON_FORMAT,
    CHARACTER_VERIFIER_PROMPT,
    TEMPORALITY_VERIFIER_PROMPT,
    ARC_DEDUPLICATOR_PROMPT
)
llm = get_llm(LLMType.INTELLIGENT)
cheap_llm = get_llm(LLMType.CHEAP)
from pydantic import BaseModel, Field

class IntermediateNarrativeArc(BaseModel):
    """Model representing an intermediate narrative arc during extraction process."""
    title: str = Field(..., description="The title of the narrative arc")
    arc_type: str = Field(..., description="Type of the arc such as 'Soap Arc'/'Genre-Specific Arc'/'Character Arc'/'Episodic Arc'/'Mythology Arc'")
    description: str = Field(..., description="A brief description of the narrative arc")
    episodic: bool = Field(..., description="If the arc is episodic or not")
    main_characters: str = Field(..., description="Main characters involved in this arc")
    interfering_episode_characters: str = Field(..., description="Interfering characters involved in this arc")
    single_episode_progression_string: str = Field("", description="The progression of the arc for the current episode")

    class Config:
        populate_by_name = True

class NarrativeArcsExtractionState(TypedDict):
    """State representation for the narrative analysis process."""
    season_analysis: str
    episode_narrative_analysis: str
    arcs_from_analysis: List[IntermediateNarrativeArc]
    arcs_from_plot: List[IntermediateNarrativeArc]
    episode_arcs: List[IntermediateNarrativeArc]  # Final arcs after verification
    present_season_arcs: List[Dict]  # Season arcs present in the episode
    file_paths: Dict[str, str]
    series: str
    season: str
    episode: str
    existing_season_entities: List[EntityLink]
    episode_plot: str
    summarized_plot: str
    season_plot: str

def initialize_state(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Initialize the state by loading necessary data from files."""
    logger.info("Initializing state with data from files.")

    # Load season analysis
    if os.path.exists(state['file_paths']['seasonal_narrative_analysis_output_path']):
        with open(state['file_paths']['seasonal_narrative_analysis_output_path'], 'r') as f:
            state['season_analysis'] = f.read()

    # Load episode narrative analysis
    if os.path.exists(state['file_paths']['episode_narrative_analysis_output_path']):
        with open(state['file_paths']['episode_narrative_analysis_output_path'], 'r') as f:
            state['episode_narrative_analysis'] = f.read()

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

def seasonal_narrative_analysis(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Analyze the season plot to identify overarching themes, character development, and narrative arcs."""
    logger.info("Starting seasonal narrative analysis.")

    # Check if the seasonal analysis output file already exists
    if len(state['season_analysis']) > 0:
        return state
    
    prompt_formatted = SEASONAL_NARRATIVE_ANALYZER_PROMPT.format(season_plot=state['season_plot'], guidelines=NARRATIVE_ARC_GUIDELINES)
    logger.debug(f"Formatted prompt: {prompt_formatted}")

    season_analysis = llm.invoke(prompt_formatted)
    logger.info("Season narrative analysis completed.")

    # Save the structured season analysis
    state['season_analysis'] = season_analysis.content.strip()

    with open(state['file_paths']['seasonal_narrative_analysis_output_path'], 'w') as f:
        f.write(state['season_analysis'])

    logger.info("Seasonal narrative analysis output saved.")

    return state

def episode_narrative_analysis(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Analyze the episode plot and provide a structured analysis based on the overall season analysis."""
    logger.info("Starting episode narrative analysis.")

    # Check if the episode narrative analysis output file already exists
    if len(state['episode_narrative_analysis']) > 0:
        return state

    episode_analysis = llm.invoke(EPISODE_NARRATIVE_ANALYZER_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        season_analysis=state['season_analysis'],
        guidelines=NARRATIVE_ARC_GUIDELINES
    ))

    logger.info("Episode narrative analysis completed.")

    # Save the structured episode analysis
    state['episode_narrative_analysis'] = episode_analysis.content.strip()

    with open(state['file_paths']['episode_narrative_analysis_output_path'], 'w') as f:
        f.write(state['episode_narrative_analysis'])

    logger.info("Episode narrative analysis output saved.")

    return state

def identify_present_season_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Identify which existing season arcs are clearly present in the current episode."""
    logger.info("Identifying present season arcs in the episode.")

    # Fetch existing season arcs from the database via NarrativeArcManager
    manager = NarrativeArcManager()
    serialized_season_arcs = []

    try:
        with manager.db_manager.session_scope() as session:
            season_arcs = manager.db_manager.get_all_narrative_arcs(series=state['series'], session=session)
            state['existing_season_arcs'] = [arc.model_dump() for arc in season_arcs]

            for arc in season_arcs:
                # Fetch progressions for this arc within the same session
                progressions = manager.db_manager.get_arc_progressions(arc.id, session=session)
                # Filter progressions for the current season
                arc.progressions = [prog for prog in progressions if prog.season == state['season']]
                if arc.progressions:  # Only include arcs that have progressions in this season
                    # Serialize the arc and its progressions
                    serialized_arc = arc.model_dump()
                    serialized_arc['progressions'] = [prog.model_dump() for prog in arc.progressions]
                    serialized_season_arcs.append(serialized_arc)

            state['season_arcs'] = serialized_season_arcs
            logger.info(f"Retrieved {len(serialized_season_arcs)} arcs for {state['series']} season {state['season']}.")

    except Exception as e:
        logger.error(f"Error managing season arcs: {e}")
        state['season_arcs'] = []

    # Prepare existing season arcs summaries
    existing_season_arcs_summaries = [
        {"title": arc['title'], "description": arc['description']}
        for arc in serialized_season_arcs if not arc['episodic']
    ]

    if len(existing_season_arcs_summaries) == 0:
        logger.warning("No existing season arcs found. Skipping present season arcs identification.")
        state['present_season_arcs'] = []
        return state

    response = llm.invoke(PRESENT_SEASON_ARCS_IDENTIFIER_PROMPT.format_messages(
        summarized_episode_plot=state['summarized_plot'],
        existing_season_arcs_summaries=json.dumps(existing_season_arcs_summaries, indent=2),
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

def extract_arcs_from_analysis(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Extract narrative arcs from the seasonal analysis and episode analysis."""
    logger.info("Extracting arcs from seasonal and episode analyses.")

    # Use present season arcs identified earlier
    present_season_arcs_summaries = [
        {"title": arc['title'], "description": arc['description']}
        for arc in state.get('present_season_arcs', [])
    ]

    response = llm.invoke(ARC_EXTRACTOR_FROM_ANALYSIS_PROMPT.format_messages(
        episode_analysis=state['episode_narrative_analysis'],
        present_season_arcs_summaries=json.dumps(present_season_arcs_summaries, indent=2),
        guidelines=NARRATIVE_ARC_GUIDELINES,
        output_json_format=OUTPUT_JSON_FORMAT
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        logger.debug(f"Cleaned LLM response: {json.dumps(arcs_data, indent=2)}")
        arcs = []
        for arc in arcs_data:
            new_arc = IntermediateNarrativeArc(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description'],
                episodic=arc['episodic'],
                main_characters=arc['main_characters'],
                interfering_episode_characters=arc['interfering_episode_characters'],
                single_episode_progression_string=arc['single_episode_progression_string']
            )
            arcs.append(new_arc)
        state['arcs_from_analysis'] = arcs
        logger.info(f"Extracted {len(arcs)} arcs from analyses.")
        logger.debug(f"Extracted arcs: {json.dumps([arc.model_dump() for arc in arcs], indent=2)}")
    except Exception as e:
        logger.error(f"Error extracting arcs from analyses: {e}")
        state['arcs_from_analysis'] = []

    return state

def extract_arcs_from_plot(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Extract narrative arcs from the episode plot, considering existing season arcs."""
    logger.info("Extracting arcs from episode plot.")

    # Use present season arcs identified earlier
    present_season_arcs_summaries = [
        {"title": arc['title'], "description": arc['description']}
        for arc in state.get('present_season_arcs', [])
    ]

    response = llm.invoke(ARC_EXTRACTOR_FROM_PLOT_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        present_season_arcs_summaries=json.dumps(present_season_arcs_summaries, indent=2),
        guidelines=NARRATIVE_ARC_GUIDELINES,
        output_json_format=OUTPUT_JSON_FORMAT
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        logger.debug(f"Cleaned LLM response: {json.dumps(arcs_data, indent=2)}")
        arcs = []
        for arc in arcs_data:
            new_arc = IntermediateNarrativeArc(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description'],
                episodic=arc['episodic'],
                main_characters=arc['main_characters'],
                interfering_episode_characters=arc['interfering_episode_characters'],
                single_episode_progression_string=arc['single_episode_progression_string']
            )
            arcs.append(new_arc)
        state['arcs_from_plot'] = arcs
        logger.info(f"Extracted {len(arcs)} arcs from plot.")
        logger.debug(f"Extracted arcs: {json.dumps([arc.model_dump() for arc in arcs], indent=2)}")
    except Exception as e:
        logger.error(f"Error extracting arcs from plot: {e}")
        state['arcs_from_plot'] = []

    return state

def verify_arc_progression(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify and adjust the progression and description of each arc."""
    logger.info("Verifying arc progressions.")

    combined_arcs = state['arcs_from_analysis'] + state['arcs_from_plot']

    verified_arcs = []
    for arc in combined_arcs:
        response = llm.invoke(ARC_PROGRESSION_VERIFIER_PROMPT.format_messages(
            episode_plot=state['episode_plot'],
            arc_to_verify=arc.model_dump(),
            output_json_format=OUTPUT_JSON_FORMAT
        ))

        try:
            verified_arc_data = clean_llm_json_response(response.content)
            if isinstance(verified_arc_data, list):
                verified_arc_data = verified_arc_data[0]

            verified_arc = IntermediateNarrativeArc(
                title=verified_arc_data['title'],
                arc_type=verified_arc_data['arc_type'],
                description=verified_arc_data['description'],
                episodic=verified_arc_data['episodic'],
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
            output_json_format=OUTPUT_JSON_FORMAT
        ))

        try:
            verified_arc_data = clean_llm_json_response(response.content)
            if isinstance(verified_arc_data, list):
                verified_arc_data = verified_arc_data[0]

            verified_arc = IntermediateNarrativeArc(
                title=verified_arc_data['title'],
                arc_type=verified_arc_data['arc_type'],
                description=verified_arc_data['description'],
                episodic=verified_arc_data['episodic'],
                main_characters=verified_arc_data['main_characters'],
                interfering_episode_characters=verified_arc_data['interfering_episode_characters'],
                single_episode_progression_string=verified_arc_data['single_episode_progression_string']
            )
            verified_arcs.append(verified_arc)
        except Exception as e:
            logger.error(f"Error verifying character roles: {e}")
            verified_arcs.append(arc)  # Keep the original arc if verification fails

    state['episode_arcs'] = verified_arcs
    logger.info(f"Verified character roles for {len(verified_arcs)} arcs.")
    return state

def verify_arc_temporality(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify and correctly categorize the temporality of each arc."""
    logger.info("Verifying arc temporality.")
    verified_arcs = []
    for arc in state['episode_arcs']:
        response = llm.invoke(TEMPORALITY_VERIFIER_PROMPT.format_messages(
            episode_plot=state['episode_plot'],
            season_plot=state['season_plot'],
            arc_to_verify=arc.model_dump(),
            output_json_format=OUTPUT_JSON_FORMAT,
            guidelines=NARRATIVE_ARC_GUIDELINES
        ))

        try:
            verified_arc_data = clean_llm_json_response(response.content)
            if isinstance(verified_arc_data, list):
                verified_arc_data = verified_arc_data[0]

            verified_arc = IntermediateNarrativeArc(
                title=verified_arc_data['title'],
                arc_type=verified_arc_data['arc_type'],
                description=verified_arc_data['description'],
                episodic=verified_arc_data['episodic'],
                main_characters=verified_arc_data['main_characters'],
                interfering_episode_characters=verified_arc_data['interfering_episode_characters'],
                single_episode_progression_string=verified_arc_data['single_episode_progression_string']
            )
            verified_arcs.append(verified_arc)
        except Exception as e:
            logger.error(f"Error verifying character roles: {e}")
            verified_arcs.append(arc)  # Keep the original arc if verification fails

    state['episode_arcs'] = verified_arcs
    logger.info(f"Verified character roles for {len(verified_arcs)} arcs.")
    return state

def verify_and_finalize_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Verify and finalize the arcs, ensuring they have the right metadata, titles, descriptions."""
    logger.info("Verifying and finalizing arcs.")

    # Combine arcs from analysis and plot
    arcs_from_analysis = state.get('arcs_from_analysis', [])
    arcs_from_plot = state.get('arcs_from_plot', [])

    combined_arcs = arcs_from_analysis + arcs_from_plot

    # Prepare arcs to verify
    arcs_to_verify = combined_arcs

    response = llm.invoke(ARC_VERIFIER_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        arcs_to_verify=arcs_to_verify,
        present_season_arcs_summaries=json.dumps(state['present_season_arcs'], indent=2),
        guidelines=NARRATIVE_ARC_GUIDELINES,
        output_json_format=OUTPUT_JSON_FORMAT
    ))

    try:
        arcs_data = clean_llm_json_response(response.content)
        arcs = []
        for arc in arcs_data:
            new_arc = IntermediateNarrativeArc(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description'],
                episodic=arc['episodic'],
                main_characters=arc['main_characters'],
                interfering_episode_characters=arc['interfering_episode_characters'],
                single_episode_progression_string=arc['single_episode_progression_string']
            )
            arcs.append(new_arc)

        state['episode_arcs'] = arcs
        logger.info(f"Verified and finalized {len(arcs)} unique arcs.")
    except Exception as e:
        logger.error(f"Error verifying arcs: {e}")
        state['episode_arcs'] = combined_arcs  # Use the combined arcs if verification fails

    return state

def deduplicate_arcs(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Deduplicate and merge similar arcs, ensuring consistent titles and descriptions."""
    logger.info("Deduplicating and merging similar arcs.")

    response = llm.invoke(ARC_DEDUPLICATOR_PROMPT.format_messages(
        episode_plot=state['episode_plot'],
        arcs_to_deduplicate=state['episode_arcs'],
        present_season_arcs_summaries=json.dumps(state['present_season_arcs'], indent=2),
        guidelines=NARRATIVE_ARC_GUIDELINES,
        output_json_format=OUTPUT_JSON_FORMAT
    ))

    try:
        deduplicated_arcs_data = clean_llm_json_response(response.content)
        deduplicated_arcs = []
        for arc in deduplicated_arcs_data:
            new_arc = IntermediateNarrativeArc(
                title=arc['title'],
                arc_type=arc['arc_type'],
                description=arc['description'],
                episodic=arc['episodic'],
                main_characters=arc['main_characters'],
                interfering_episode_characters=arc['interfering_episode_characters'],
                single_episode_progression_string=arc['single_episode_progression_string']
            )
            deduplicated_arcs.append(new_arc)
        state['episode_arcs'] = deduplicated_arcs
        logger.info(f"Deduplicated to {len(deduplicated_arcs)} unique arcs.")
    except Exception as e:
        logger.error(f"Error deduplicating arcs: {e}")
        # Keep original arcs if deduplication fails

    return state

def save_episode_arcs_to_json(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Save the episode arcs with IDs and their corresponding progressions to a JSON file."""
    logger.info("Saving episode arcs with IDs and progressions to JSON file.")

    if 'episode_narrative_arcs_path' in state['file_paths']:
        episode_arcs = []
        for arc in state['episode_arcs']:
            arc.main_characters = normalize_names(arc.main_characters, state['existing_season_entities'], cheap_llm)
            arc.interfering_episode_characters = normalize_names(arc.interfering_episode_characters, state['existing_season_entities'], cheap_llm)
            # Keep only the progression for the current episode
            current_progression = next((p for p in arc.progressions if p.episode == state['episode'] and p.season == state['season']), None)
            if current_progression:
                arc_dict = arc.model_dump()
                arc_dict['progressions'] = [current_progression.model_dump()]
                episode_arcs.append(arc_dict)

        with open(state['file_paths']['episode_narrative_arcs_path'], 'w') as f:
            json.dump(episode_arcs, f, indent=2)
        logger.info("Episode narrative arcs with IDs and progressions saved.")

    return state

def save_season_arcs_to_json(state: NarrativeArcsExtractionState) -> NarrativeArcsExtractionState:
    """Save the season arcs with all progressions to a JSON file."""
    logger.info("Saving season arcs with all progressions to JSON file.")

    if 'season_narrative_arcs_path' in state['file_paths']:
        season_arcs_data = []
        for arc in state['season_arcs']:
            arc_dict = arc.model_dump()
            # Include all progressions for the current season
            arc_dict['progressions'] = [
                prog.model_dump() for prog in arc.progressions 
                if prog.season == state['season']
            ]
            season_arcs_data.append(arc_dict)

        with open(state['file_paths']['season_narrative_arcs_path'], 'w') as f:
            json.dump(season_arcs_data, f, indent=2)
        logger.info("Season narrative arcs with all progressions saved.")

    return state

def create_narrative_arc_graph():
    """Create and configure the state graph for narrative arc extraction."""
    workflow = StateGraph(NarrativeArcsExtractionState)

    # Add all existing nodes
    workflow.add_node("initialize_state_node", initialize_state)
    workflow.add_node("season_analysis_node", seasonal_narrative_analysis)
    workflow.add_node("episode_analysis_node", episode_narrative_analysis)
    workflow.add_node("identify_present_season_arcs_node", identify_present_season_arcs)
    workflow.add_node("extract_arcs_from_analysis_node", extract_arcs_from_analysis)
    workflow.add_node("extract_arcs_from_plot_node", extract_arcs_from_plot)
    workflow.add_node("verify_arc_progression_node", verify_arc_progression)
    workflow.add_node("verify_character_roles_node", verify_character_roles)
    workflow.add_node("verify_arc_temporality_node", verify_arc_temporality)
    workflow.add_node("verify_and_finalize_arcs_node", verify_and_finalize_arcs)
    workflow.add_node("deduplicate_arcs_node", deduplicate_arcs)  # Add new node

    # Set the entry point and edges
    workflow.set_entry_point("initialize_state_node")
    workflow.add_edge("initialize_state_node", "season_analysis_node")
    workflow.add_edge("season_analysis_node", "episode_analysis_node")
    workflow.add_edge("episode_analysis_node", "identify_present_season_arcs_node")
    workflow.add_edge("identify_present_season_arcs_node", "extract_arcs_from_analysis_node")
    workflow.add_edge("extract_arcs_from_analysis_node", "extract_arcs_from_plot_node")
    workflow.add_edge("extract_arcs_from_plot_node", "verify_arc_progression_node")
    workflow.add_edge("verify_arc_progression_node", "verify_character_roles_node")
    workflow.add_edge("verify_character_roles_node", "verify_arc_temporality_node")
    workflow.add_edge("verify_arc_temporality_node", "verify_and_finalize_arcs_node")
    workflow.add_edge("verify_and_finalize_arcs_node", "deduplicate_arcs_node")  # Add new edge
    workflow.add_edge("deduplicate_arcs_node", END)

    logger.info("Narrative arc graph workflow created successfully.")

    return workflow.compile()

def extract_narrative_arcs(file_paths: Dict[str, str], series: str, season: str, episode: str) -> None:
    """Extract narrative arcs from the provided file paths and save the results to a JSON file."""
    logger.info("Starting extract_narrative_arcs function")
    graph = create_narrative_arc_graph()

    initial_state = NarrativeArcsExtractionState(
        season_analysis="",
        episode_narrative_analysis="",
        arcs_from_analysis=[],
        arcs_from_plot=[],
        episode_arcs=[],
        present_season_arcs=[],
        file_paths=file_paths,
        series=series,
        season=season,
        episode=episode,
        existing_season_entities=[],
        episode_plot="",
        summarized_plot="",
        season_plot=""
    )
    logger.info("Invoking the graph")
    result = graph.invoke(initial_state)
    logger.info("Graph execution completed")

    # Save the results to a JSON file
    suggested_arcs = [arc.model_dump() for arc in result['episode_arcs']]
    
    save_json(suggested_arcs, file_paths['suggested_episode_arc_path'])

    logger.info(f"Suggested episode arcs saved to {file_paths['suggested_episode_arc_path']}")